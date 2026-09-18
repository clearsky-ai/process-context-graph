"""LLM sequence prediction of the activities an email reports.

This is the sequence-prediction evolution of ``classify_emails.classify_multi``:
instead of a SET of activities, the model returns the ORDERED LIST of activities
the sender performed, as reported by one email (the email may cover several
grouped events). Order is preserved and DUPLICATES ARE KEPT - a real trace can
repeat an activity, and every predicted activity becomes one reconstructed OCEL
event downstream.

The user prompt is a unified prefix (catalog) plus up to two independent source
blocks — mail and OCEL timeline — each with its own instructions. Either may be
absent. Graph context belongs to the agentic path (``agent_predictor``), which
never comes through here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import time

from src.azure_client import make_azure_openai_client, raise_if_fatal_azure_error
from src.prompt_store import P, render

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = P("system")

# Bound the completion size. Predictions are tiny JSON, but gpt-5.x reasoning
# models also spend part of this budget on hidden reasoning, so keep it modest
# yet generous. Capping it (vs the model default) reduces the tokens-per-minute
# pressure that triggers 429s. Override with RECON_MAX_COMPLETION_TOKENS.
_MAX_COMPLETION_TOKENS = int(os.getenv("RECON_MAX_COMPLETION_TOKENS", "2000"))

# Reasoning budget for gpt-5.x reasoning models ("minimal" | "low" | "medium" |
# "high"). Empty means "do not send the parameter" (the model default), which is
# also the escape hatch for deployments that reject it. Override with
# RECON_REASONING_EFFORT.
_DEFAULT_REASONING_EFFORT = os.getenv("RECON_REASONING_EFFORT", "high").strip()

# Rate-limit (429) backoff: a 429 under high request volume is transient, so we
# wait and retry instead of aborting the whole run. Waits honor a Retry-After
# header when present, else grow exponentially (capped), with jitter added.
_RATE_LIMIT_MAX_RETRIES = int(os.getenv("RECON_RATE_LIMIT_RETRIES", "6"))
_RATE_LIMIT_BASE_SLEEP = float(os.getenv("RECON_RATE_LIMIT_BASE_SLEEP", "5"))
_RATE_LIMIT_MAX_SLEEP = float(os.getenv("RECON_RATE_LIMIT_MAX_SLEEP", "60"))

# Cross-case dedup cache: an identical prediction prompt (same email + catalog +
# K cap + single-activity mode) always yields the same prediction, so predict
# each distinct email ONCE and reuse it. In the pattern experiment the same
# shared (non-pattern) email is reconstructed at every absorption level; this
# skips re-calling the LLM for it. Cases run sequentially (only emails within a
# case run in parallel, and those prompts are all distinct), so there is no
# cross-level race. Disable with RECON_CACHE=0.
_CACHE_ENABLED = os.getenv("RECON_CACHE", "1").lower() not in (
    "0", "false", "no", "off",
)
# activities, confidence, reasoning, raw assistant text
_PREDICT_CACHE: dict[str, tuple[list[str], float, str, str]] = {}


def _predict_cache_key(
    system_prompt: str, user_prompt: str, max_activities: int, force_single: bool,
    reasoning_effort: str,
) -> str:
    """Stable hash of everything that determines a prediction's final result."""
    payload = (
        f"{max_activities}\x00{int(force_single)}\x00{reasoning_effort}\x00"
        f"{system_prompt}\x00{user_prompt}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_rate_limit_error(exc: BaseException) -> bool:
    """True if *exc* is an Azure/OpenAI 429 rate-limit error (transient)."""
    try:
        import openai

        if isinstance(exc, openai.RateLimitError):
            return True
    except Exception:
        pass
    msg = str(exc).lower()
    return any(
        marker in msg
        for marker in ("429", "rate limit", "ratelimit",
                       "too_many_requests", "rate_limit_exceeded")
    )


def _retry_after_seconds(exc: BaseException, default: float) -> float:
    """Seconds to wait before a retry, from a Retry-After header or *default*."""
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None)
    get = getattr(headers, "get", None)
    if callable(get):
        val = get("retry-after") or get("Retry-After")
        if val:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    return default


def build_catalog(schema: list[dict]) -> str:
    """Compact, LLM-friendly listing of the activity classes."""
    return "\n".join(
        f"- {e['event_type']}: {e['description']}" for e in schema
    )


def build_allowed(schema: list[dict]) -> dict[str, str]:
    """Map lowercased event_type -> canonical event_type for normalization."""
    return {e["event_type"].lower(): e["event_type"] for e in schema}


def build_user_prompt(
    subject: str,
    body: str,
    sender: str,
    objects: str,
    catalog: str,
    ocel_text: str = "",
    force_single_activity: bool = False,
    max_activities: int = 0,
) -> str:
    """Build the user prompt: catalog prefix + present source blocks.

    Each of mail / OCEL is omitted when its text is empty. When
    *force_single_activity* is True the model is asked for EXACTLY ONE
    activity; when *max_activities* > 0 (and not single mode) the model is
    told to return AT MOST that many activities (the length cap K).
    """
    cap_block = (
        render(P("cap_block"), n=max_activities,
               word="activity" if max_activities == 1 else "activities")
        if (max_activities and not force_single_activity)
        else ""
    )
    mail_block = (
        render(
            P("mail_block"),
            sender=sender,
            objects=objects,
            subject=subject,
            body=body,
        ) + "\n\n"
        if any(str(v).strip() for v in (sender, objects, subject, body))
        else ""
    )
    ocel_block = (
        render(P("ocel_block"), timeline=ocel_text) + "\n\n"
        if str(ocel_text).strip()
        else ""
    )
    if force_single_activity:
        task_block = P("task_single")
    else:
        task_block = P("task_multi") + cap_block
    return render(
        P("main"),
        catalog=catalog,
        mail_block=mail_block,
        ocel_block=ocel_block,
        task_block=task_block,
    )


class SequencePredictor:
    """Predict the ordered activity sequence an email reports, via Azure OpenAI.

    When *force_single_activity* is True the predictor asks for and returns
    exactly one activity per email (single-span mode); the default (False)
    reconstructs the full ordered sequence.

    *reasoning_effort* is the gpt-5.x reasoning budget sent with every call;
    an empty value omits the parameter and leaves the model default in place.
    """

    def __init__(
        self,
        force_single_activity: bool = False,
        reasoning_effort: str = _DEFAULT_REASONING_EFFORT,
    ) -> None:
        self._client = make_azure_openai_client()
        self._deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        self._force_single_activity = force_single_activity
        self._reasoning_effort = (reasoning_effort or "").strip()

    def _create_with_backoff(self, messages: list[dict]):
        """Call chat.completions, retrying on 429 rate limits with backoff.

        A 429 under high concurrency is transient, so we wait (honoring a
        Retry-After header when present, else exponential backoff with jitter,
        capped) and retry. Any non-rate-limit error, or a 429 that survives all
        retries, propagates to the caller so a truly broken client still aborts.
        """
        extra = (
            {"reasoning_effort": self._reasoning_effort}
            if self._reasoning_effort else {}
        )
        for attempt in range(_RATE_LIMIT_MAX_RETRIES + 1):
            try:
                return self._client.chat.completions.create(
                    model=self._deployment,
                    messages=messages,
                    response_format={"type": "json_object"},
                    max_completion_tokens=_MAX_COMPLETION_TOKENS,
                    **extra,
                )
            except Exception as exc:
                if not _is_rate_limit_error(exc) or attempt >= _RATE_LIMIT_MAX_RETRIES:
                    raise
                fallback = min(_RATE_LIMIT_BASE_SLEEP * (2 ** attempt), _RATE_LIMIT_MAX_SLEEP)
                delay = _retry_after_seconds(exc, fallback) + random.uniform(0, 1)
                logger.warning(
                    "reconstruction rate-limited (429); backing off %.1fs "
                    "(attempt %d/%d)",
                    delay, attempt + 1, _RATE_LIMIT_MAX_RETRIES,
                )
                time.sleep(delay)
        raise RuntimeError("rate-limit retry loop exited unexpectedly")

    def predict(
        self,
        subject: str,
        body: str,
        sender: str,
        objects: str,
        catalog: str,
        allowed: dict[str, str],
        ocel_text: str = "",
        max_activities: int = 0,
    ) -> tuple[list[str], float, str, dict]:
        """Return (ordered_activities, confidence, reasoning, llm_call).

        Each activity is normalized to an exact catalog value; values not in the
        catalog are dropped. ORDER is preserved and duplicates are kept. When
        *max_activities* > 0 the returned list is hard-truncated to that length
        (the K cap) - over-prediction is bounded; under-prediction is left as-is.
        *llm_call* is the system/user/assistant exchange for prompt markdown.
        """
        user_prompt = build_user_prompt(
            subject, body, sender, objects, catalog,
            ocel_text=ocel_text,
            force_single_activity=self._force_single_activity,
            max_activities=max_activities,
        )
        call: dict = {
            "pass": "predict",
            "ok": True,
            "cached": False,
            "system": _SYSTEM_PROMPT,
            "user": user_prompt,
            "assistant": None,
            "error": None,
            "max_completion_tokens": _MAX_COMPLETION_TOKENS,
            "reasoning_effort": self._reasoning_effort,
        }
        cache_key = (
            _predict_cache_key(
                _SYSTEM_PROMPT, user_prompt, max_activities,
                self._force_single_activity, self._reasoning_effort,
            )
            if _CACHE_ENABLED else None
        )
        if cache_key is not None:
            cached = _PREDICT_CACHE.get(cache_key)
            if cached is not None:
                activities, confidence, reasoning, raw = cached
                call["cached"] = True
                call["assistant"] = raw
                call["note"] = "Reused from in-process cache (no new LLM call)."
                return list(activities), confidence, reasoning, call
        response = self._create_with_backoff([
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ])
        raw = (response.choices[0].message.content or "").strip()
        call["assistant"] = raw
        stripped = raw
        if stripped.startswith("```"):
            stripped = stripped.split("```")[1]
            if stripped.startswith("json"):
                stripped = stripped[4:]
        try:
            parsed = json.loads(stripped.strip())
        except Exception as exc:
            call["ok"] = False
            call["error"] = str(exc)
            return [], 0.0, f"error: {exc}", call

        raw_activities = parsed.get("activities", [])
        if isinstance(raw_activities, str):
            raw_activities = [raw_activities]
        confidence = float(parsed.get("confidence", 0) or 0)
        reasoning = str(parsed.get("reasoning", "")).strip()

        activities: list[str] = []
        for value in raw_activities:
            canonical = allowed.get(str(value).strip().lower(), "")
            if canonical:  # keep order AND duplicates
                activities.append(canonical)
        if self._force_single_activity:  # keep only the single best activity
            activities = activities[:1]
        elif max_activities and len(activities) > max_activities:
            activities = activities[:max_activities]  # hard K cap (bounded FP)
        if cache_key is not None:
            _PREDICT_CACHE[cache_key] = (
                list(activities), confidence, reasoning, call["assistant"] or "",
            )
        return activities, confidence, reasoning, call

    def predict_safe(
        self,
        subject: str,
        body: str,
        sender: str,
        objects: str,
        catalog: str,
        allowed: dict[str, str],
        ocel_text: str = "",
        max_activities: int = 0,
    ) -> tuple[list[str], float, str, dict]:
        """Like :meth:`predict` but converts fatal Azure errors and re-raises them.

        A content-level failure (bad JSON on one email) returns an empty
        prediction plus the captured prompt exchange; a broken client
        (auth/rate/quota/deployment) is fatal and propagated so the caller can
        abort the whole run.
        """
        try:
            return self.predict(
                subject, body, sender, objects, catalog, allowed,
                ocel_text=ocel_text,
                max_activities=max_activities,
            )
        except Exception as exc:
            raise_if_fatal_azure_error(exc)
            user_prompt = build_user_prompt(
                subject, body, sender, objects, catalog,
                ocel_text=ocel_text,
                force_single_activity=self._force_single_activity,
                max_activities=max_activities,
            )
            return [], 0.0, f"error: {exc}", {
                "pass": "predict",
                "ok": False,
                "cached": False,
                "system": _SYSTEM_PROMPT,
                "user": user_prompt,
                "assistant": None,
                "error": str(exc),
                "max_completion_tokens": _MAX_COMPLETION_TOKENS,
                "reasoning_effort": self._reasoning_effort,
            }
