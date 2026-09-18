"""Agentic reconstruction of a hidden email's activity sequence (AutoGen v0.4).

Used only for the graph-enabled configs (``use_graph=true``). For each hidden
email an :class:`~autogen_agentchat.agents.AssistantAgent` (GPT-5.4 via Azure)
reconstructs the ordered activities the email reports by calling four retrieval
tools over the context graph and the corrupted OCELs (see :mod:`src.tools`),
then returns a single JSON object ``{"activities", "confidence", "reasoning"}``.

The model is wired through Azure OpenAI reusing the exact auth precedence of
:mod:`src.azure_client` (API key if present, else a ``DefaultAzureCredential``
bearer token). GPT-5.4 is the Azure *deployment* target
(``AZURE_OPENAI_DEPLOYMENT``); the ``model`` name is only used by AutoGen to look
up capabilities, so an explicit ``model_info`` is supplied for the unknown name.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import random

from src.azure_client import _DEFAULT_API_VERSION, _token_provider
from src.tools import build_tools

logger = logging.getLogger(__name__)

# Bound how long we wait out transient 429s per email before giving up.
_RATE_LIMIT_RETRIES = int(os.getenv("RECON_RATE_LIMIT_RETRIES", "6"))
_RATE_LIMIT_BASE_SLEEP = float(os.getenv("RECON_RATE_LIMIT_BASE_SLEEP", "5"))
_RATE_LIMIT_MAX_SLEEP = float(os.getenv("RECON_RATE_LIMIT_MAX_SLEEP", "60"))

# Reasoning budget for gpt-5.x reasoning models ("minimal" | "low" | "medium" |
# "high"). Empty means "do not send the parameter" (the model default).
_DEFAULT_REASONING_EFFORT = os.getenv("RECON_REASONING_EFFORT", "high").strip()

# The system prompt is assembled from three segments so the retrieval-tool
# guidance can be dropped when the agent runs single-shot on its cold-start
# context (use_agentic_tools=false). The intro and _sys_decide are shared;
# the tools segment is included only when tools are bound.
_SYS_PREAMBLE = (
    "You are an expert process-mining analyst for a loan-origination process. "
    "Your job is to REVERSE-ENGINEER the ORDERED list of OCEL activities that ONE "
    "hidden email reports. The hidden email is the internal_email_sent event in "
    "the current application's OCEL; the activities it hides are exactly what you "
    "must output - no more, no fewer, in the order they happened.\n\n"
    "THE CONTEXT GRAPH - understand its ontology; it was extracted from this "
    "case's messages and process log:\n"
)

_SYS_ONTOLOGY = (
    "- ActivityCluster: the unnamed group of activities this case ran. It is a "
    "placeholder - the graph records that the activities happened and what they "
    "touched, never which ones they were.\n"
    "- ProcessInstance: the loan application/case (its OCEL case object) that "
    "the cluster is part of.\n"
    "- DecisionNode: a decision taken in the case (decision_type, final_choice, "
    "outcome_status) - a choice point.\n"
    "- RationaleNode: the justification (the WHY) behind a decision.\n"
    "- EvidenceNode: a verbatim span from a message/attachment (signal_source) "
    "that a decision rests on or the case produced.\n"
    "- Actor: a person/team.\n"
)

# The only wording that tracks a knob: with mask_execution_verbs on, the Actor
# edges into the cluster all render as linked_to, so the gloss must say that
# rather than list verbs the agent will never see.
_SYS_ACTOR_EDGE_MASKED = "Actor -linked_to-> ActivityCluster; "
_SYS_ACTOR_EDGE_PLAIN = (
    "Actor -prepared/reviewed/ran/completed/approved/signed_off-> "
    "ActivityCluster; "
)


def _sys_intro(mask_execution_verbs: bool) -> str:
    """Ontology gloss, matched to how the subgraph will actually be rendered."""
    actor_edge = (
        _SYS_ACTOR_EDGE_MASKED if mask_execution_verbs else _SYS_ACTOR_EDGE_PLAIN
    )
    return (
        _SYS_PREAMBLE + _SYS_ONTOLOGY
        + "Key edges: ActivityCluster -has_decision-> DecisionNode; DecisionNode "
        "-justified_by-> RationaleNode; DecisionNode -supported_by-> EvidenceNode; "
        "ActivityCluster -produced-> EvidenceNode; EvidenceNode -explained_by-> "
        "RationaleNode; Actor -made_decision-> DecisionNode; DecisionNode "
        "-acted_on-> ActivityCluster; " + actor_edge
        + "ActivityCluster -part_of-> ProcessInstance.\n"
        "KEY INSIGHT: the activities themselves are never named in the graph - "
        "naming them IS your task. The ActivityCluster marks where they sit; the "
        "decisions, rationales, and evidence hanging off it are the TRACES they "
        "left. Do not just read the EvidenceNode text; work out from the whole "
        "cluster which activities must have occurred.\n\n"
    )


def _sys_tools() -> str:
    return (
        "TOOLS - gather comparative evidence before answering:\n"
        "- find_similar_decisions(decision_node_id, k, hops) / "
        "find_similar_rationals(rationale_id, k, hops): hybrid semantic+structural "
        "nearest nodes, each with its neighborhood and owning application. START "
        "HERE, with a seed id from the task.\n"
        "- get_ocel_of_application(application_id): an application's observed OCEL "
        "timeline (its real event log); that case's hidden activities sit behind its "
        "internal_email_sent event. Use it on a COMPARABLE application returned by "
        "the similarity tools, to see how that case's log names the steps a case in "
        "this situation goes through. Calling it on the CURRENT application is "
        "pointless - that timeline is already quoted in the task.\n"
        "- get_application_context_subgraph(application_id, hops): the subgraph "
        "around an application's ActivityCluster - use it to read a comparable "
        "case's evidence spans next to its log.\n"
        "A comparable case is the only place you can learn which catalog activity a "
        "given kind of span corresponds to, so look at one before you commit.\n\n"
    )


def _sys_decide(use_tools: bool) -> str:
    gather = (
        "Gather comparative evidence with the tools, then walk"
        if use_tools
        else "Walk"
    )
    return (
        "HOW TO DECIDE - one span at a time:\n"
        "Every span=<n> EvidenceNode came from THIS hidden email, and the subgraph "
        f"lists them in the order the sender narrates them. {gather} the spans in "
        "ascending order and give each one a verdict: NO STEP, STEP, or MULTI. A "
        "span yields 0, 1 or several activities accordingly.\n\n"
        "STEP - the span reports one event. Name the single catalog activity it maps "
        "to, copied verbatim. A span is a STEP when it asserts that work WAS "
        "performed (a completed action, in the first or the third person) OR that "
        "the case, an offer or the paperwork HAS REACHED a state (a present-tense "
        "assertion of the form \"<subject> is now <state>\"). Those state "
        "assertions are exactly what the A_ and O_ activities are, so never dismiss "
        "one for being passively phrased.\n\n"
        "MULTI - one span reports SEVERAL events and yields one activity for each. "
        "This takes TWO separate assertions inside that one span: that an action "
        "was carried out, AND that the state it produced now holds. Where the "
        "catalog defines those as two DIFFERENT activities, emit BOTH, in clause "
        "order. A span that asserts only the work, or only the state, is a STEP and "
        "yields ONE activity - the catalog happening to hold a related activity is "
        "NOT by itself a reason to emit two. A trailing clause that merely "
        "elaborates the state, naming what is being checked, awaited or worked "
        "through, restates that one event rather than adding a second. Two "
        "clauses count as two events only when both are main clauses. An action "
        "named in a subordinate clause - one opening with After, Once, Having or "
        "While and carrying a verb in its -ing form - is background for the main "
        "clause, and that span yields only what the main clause reports.\n\n"
        "NO STEP - the span reports no event and yields no activity. Only three "
        "things qualify:\n"
        "- its point is a step that has NOT happened yet: future tense, a stated "
        "intention, or an explicit \"next step\" framing;\n"
        "- it is a request, a pleasantry or a sign-off;\n"
        "- it restates an action another span already reported.\n\n"
        "MATCH MEANINGS, NOT WORDING. A state assertion is a STEP only when the state "
        "it asserts is what some catalog activity MEANS - read the definition, do not "
        "go by which name sounds closest. If a span says only that the case is queued, "
        "waiting, or ready to be looked at, and no catalog activity means that, it is "
        "NO STEP. Equally, judge each span on its own assertion: a state the case has "
        "already reached stays a STEP even when the sentence goes on to mention what "
        "comes next.\n\n"
        "THEN:\n"
        "- WHO acted is irrelevant. A span that credits a colleague by name "
        "(\"<colleague> picked it up\") is still a step THIS email reports; never skip "
        "a span for being in the third person, and never mistake a named colleague "
        "for another email's context.\n"
        "- ONE action = ONE activity, but ONE SPAN may carry more than one action. "
        "When a span asserts the work AND, in a separate clause, that the resulting "
        "state now holds, and the catalog lists those as two activities, emit both "
        "rather than choosing. When the span asserts only one of them - only the "
        "work, or only the state - emit only that one. Collapse to a single "
        "activity when the span says the same thing twice. W_ activities are "
        "first-class: emit one whenever a span reports that work being done.\n"
        "- Two spans describing the SAME action are one activity; repeat an activity "
        "only when it genuinely happened twice.\n"
        "- Emit the activities in STRICTLY ASCENDING span order, and within a MULTI "
        "span in clause order. Never reorder by timestamp, by the catalog's listing "
        "order, or by what usually happens next.\n"
        "- Before answering, re-read the spans you called NO STEP, and re-check "
        "whether a span you called STEP was really a MULTI. Dropping a real step "
        "(the first one especially) and padding with a step that has not happened "
        "yet are the two ways to fail here, and they cost the same.\n\n"
        "THE OBSERVED TIMELINE constrains, it does not generate. Use it to tell "
        "look-alike activities apart. Never add an activity because the timeline or "
        "the usual process makes it plausible - every activity must come from a STEP "
        "or MULTI span. The hidden email's events were removed from that timeline, "
        "so an activity still visible there belongs to a different event unless a "
        "span reports it.\n\n"
        "Always pick at least one activity; never return an empty list. When "
        "finished, reply with a SINGLE JSON object and NOTHING else:\n"
        '{"activities": ["<exact event_type>", ...], "confidence": <number 0-1>, '
        '"reasoning": "<one short sentence>"}'
    )


def build_system_prompt(
    use_tools: bool = True, mask_execution_verbs: bool = True,
) -> str:
    """Assemble the reconstructor system prompt.

    With *use_tools* the retrieval-tool guidance is included; without it the
    agent answers single-shot from the cold-start context in the task prompt.
    *mask_execution_verbs* must match how the subgraph is rendered, or the
    ontology gloss names edges the agent never sees.
    """
    return (
        _sys_intro(mask_execution_verbs)
        + (_sys_tools() if use_tools else "")
        + _sys_decide(use_tools)
    )


_SIMILARITY_TOOLS = frozenset({"find_similar_decisions", "find_similar_rationals"})

# Appended to the task and re-run once when the agent answered without ever
# consulting a comparable case. Self-contained: the retry gets a fresh agent.
_TOOL_NUDGE = (
    "You answered without consulting a comparable case, so your mapping from "
    "spans to catalog activities rests on nothing but the wording of this one "
    "email. Do it properly now:\n"
    "1. Call find_similar_decisions or find_similar_rationals on one of the seed "
    "ids listed above.\n"
    "2. Call get_ocel_of_application on a COMPARABLE application from those "
    "results - never on the current application, whose timeline you already "
    "have - and read how its observed log names the steps.\n"
    "3. Then build the span ledger (label every span NO STEP, STEP or MULTI, "
    "including spans that credit a colleague by name) and answer."
)


def _is_rate_limit_error(exc: BaseException) -> bool:
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


def _extract_json_object(text: str) -> "dict | None":
    """Best-effort parse of the last JSON object in *text*."""
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t[:4].lower() == "json":
            t = t[4:]
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(t[start:end + 1])
    except json.JSONDecodeError:
        return None


def _render_content(content) -> str:
    """Render a message's content (str, or list of tool calls/results)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            name = getattr(item, "name", None)
            arguments = getattr(item, "arguments", None)
            result = getattr(item, "content", None)
            if name is not None and arguments is not None:
                parts.append(f"CALL {name}({arguments})")
            elif name is not None and result is not None:
                parts.append(f"RESULT {name}:\n{result}")
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


class AgenticPredictor:
    """Reconstruct hidden emails with a tool-using AutoGen agent."""

    def __init__(
        self,
        graph_index,
        ocel_store,
        *,
        model: str = "gpt-5.4",
        default_k: int = 5,
        default_hops: int = 1,
        max_tool_iterations: int = 6,
        context_hops: int = 2,
        require_similarity_lookup: bool = True,
        use_tools: bool = True,
        reasoning_effort: str = _DEFAULT_REASONING_EFFORT,
        mask_execution_verbs: bool = True,
    ) -> None:
        self._graph_index = graph_index
        self._ocel_store = ocel_store
        self._model = model
        self._default_k = default_k
        self._default_hops = default_hops
        self._max_tool_iterations = max(1, int(max_tool_iterations))
        self._context_hops = context_hops
        # When false, the agent answers single-shot from its cold-start context
        # (case subgraph + timeline) with NO retrieval tools and no lookup retry.
        self._use_tools = bool(use_tools)
        # When the agent answers without ever calling a similarity tool, re-ask
        # once with an explicit protocol. Costs a second agent run only on the
        # cases that skipped the lookup. Meaningless (and disabled) without tools.
        self._require_similarity_lookup = bool(require_similarity_lookup) and self._use_tools
        # Must match the GraphIndex this predictor renders through, or the
        # ontology gloss names a node type the agent never sees.
        self._mask_execution_verbs = bool(mask_execution_verbs)
        # System prompt drops the tool guidance when running tool-free.
        self._system_prompt = build_system_prompt(
            self._use_tools, self._mask_execution_verbs,
        )
        self._reasoning_effort = (reasoning_effort or "").strip()
        self._client = self._make_model_client(model, self._reasoning_effort)

    # ------------------------------------------------------------------ #
    #  Model client                                                       #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _make_model_client(model: str, reasoning_effort: str = ""):
        from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

        try:
            from autogen_core.models import ModelFamily
            family = ModelFamily.UNKNOWN
        except Exception:  # pragma: no cover - fallback for older layouts
            family = "unknown"

        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise RuntimeError("AZURE_OPENAI_ENDPOINT is not set.")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        if not deployment:
            raise RuntimeError("AZURE_OPENAI_DEPLOYMENT is not set.")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", _DEFAULT_API_VERSION)

        model_info = {
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": family,
            "structured_output": True,
        }
        common = dict(
            model=model,
            azure_endpoint=endpoint,
            azure_deployment=deployment,
            api_version=api_version,
            model_info=model_info,
        )
        if reasoning_effort:
            common["reasoning_effort"] = reasoning_effort
        api_key = os.getenv("AZURE_OPENAI_API_KEY")

        def _build(**extra):
            if api_key:
                return AzureOpenAIChatCompletionClient(api_key=api_key, **extra)
            return AzureOpenAIChatCompletionClient(
                azure_ad_token_provider=_token_provider(), **extra
            )

        try:
            return _build(**common)
        except (TypeError, ValueError) as exc:
            if not reasoning_effort:
                raise
            # Older AutoGen builds reject unknown create arguments; fall back to
            # the model default rather than failing the whole run.
            logger.warning(
                "AutoGen client rejected reasoning_effort=%r (%s); continuing "
                "with the model default.", reasoning_effort, exc,
            )
            common.pop("reasoning_effort", None)
            return _build(**common)

    def _build_agent(self, current_app: str | None = None):
        from autogen_agentchat.agents import AssistantAgent

        # Tool-free mode: no retrieval tools, so the agent answers single-shot
        # from the cold-start context already in the task prompt.
        tools = (
            build_tools(
                self._graph_index, self._ocel_store,
                default_k=self._default_k, default_hops=self._default_hops,
                current_app=current_app,
            )
            if self._use_tools
            else []
        )
        kwargs = dict(
            name="reconstructor",
            model_client=self._client,
            tools=tools,
            system_message=self._system_prompt,
            reflect_on_tool_use=True,
        )
        # max_tool_iterations enables dependent multi-step tool use; guard for
        # older AutoGen builds that lack it. Irrelevant when no tools are bound.
        if (
            self._use_tools
            and "max_tool_iterations" in inspect.signature(AssistantAgent.__init__).parameters
        ):
            kwargs["max_tool_iterations"] = self._max_tool_iterations
        return AssistantAgent(**kwargs)

    # ------------------------------------------------------------------ #
    #  Prompt assembly                                                    #
    # ------------------------------------------------------------------ #
    def _build_task(
        self,
        application_id: str,
        catalog: str,
        ocel_text: str,
        seeds: dict,
    ) -> str:
        subgraph = self._graph_index.application_subgraph_text(
            application_id, self._context_hops, this_email=True
        )

        lines = [
            "Reconstruct the ordered activities reported by the hidden email in "
            f"application {application_id}.",
            "",
            "Activity catalog (choose from these; copy verbatim). Each line is "
            '"<event_type>: <description>"; the description explains MEANING, not '
            "order:",
            catalog,
            "",
            f"Current application id: {application_id}",
        ]
        if self._use_tools:
            lines += [
                "Seed node ids in this application (use them with the similarity tools):",
                f"- decisions: {', '.join(seeds['decisions']) or '(none)'}",
                f"- rationales: {', '.join(seeds['rationales']) or '(none)'}",
            ]
        lines += [
            "",
            "The subgraph below is this case in the ontology - its ActivityCluster "
            "plus the decisions, rationales, evidence spans and relationships "
            "that hang off it, all extracted from THIS hidden email. Reason over "
            "the WHOLE subgraph (not just the evidence text), then build your span "
            "ledger from the span=<n> EvidenceNodes: label every span NO STEP, STEP "
            "or MULTI, including spans that credit a colleague by name, and map it "
            "to that many catalog activities. Only a span that asserts an action "
            "AND, in a separate clause, the state that action produced is a MULTI; "
            "a span asserting just one of the two is a STEP yielding one activity.",
            subgraph,
        ]
        if ocel_text.strip():
            lines += [
                "",
                "Observed case timeline. Lines BEFORE '>>> THIS' are "
                "already-observed PAST events and lines AFTER are FUTURE observed "
                "events - do NOT output any of them. Use the timeline ONLY to "
                "disambiguate look-alike activities; it does NOT set your output "
                "order (that comes from the email's span order):",
                ocel_text,
            ]
        if self._use_tools:
            lines += [
                "",
                "TOOL PROTOCOL for this case - the timeline above is already the "
                "current application's, so do NOT call get_ocel_of_application on "
                f"{application_id}; it would tell you nothing new. Instead:",
                "1. Call find_similar_decisions or find_similar_rationals on one of "
                "the seed ids above to find comparable cases.",
                "2. Call get_ocel_of_application (and, if useful, "
                "get_application_context_subgraph) on a COMPARABLE application from "
                "those results, to see how an observed log names the steps of a case "
                "in this situation.",
                "3. Only then build your span ledger and answer.",
            ]
        else:
            lines += [
                "",
                "Build your span ledger from the subgraph and timeline above and "
                "answer directly - no external lookups are available for this case.",
            ]
        lines += [
            "",
            "Respond with a SINGLE JSON object and nothing else: "
            '{"activities": ["<exact event_type>", ...], "confidence": '
            '<number 0-1>, "reasoning": "<one short sentence>"}',
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Prediction                                                         #
    # ------------------------------------------------------------------ #
    async def predict(
        self,
        record: dict,
        catalog: str,
        allowed: dict,
        ocel_text: str = "",
    ) -> tuple[list[str], float, str, dict]:
        """Return (activities, confidence, reasoning, llm_call).

        ``llm_call`` mirrors the shape the non-agentic path logs, with an extra
        ``transcript`` list capturing the full agent run (messages + tool calls)
        for the per-email markdown.
        """
        # The context graph keys ProcessInstance nodes by the graph_key
        # (e.g. "application_<n>__<opaque token>") that prepare_graph_inputs /
        # context_graph_ingest use, NOT the corrupted OCEL case id
        # ("Application_<n>__<opaque token>"). Use graph_key so the graph tools
        # resolve the application; OcelStore accepts graph_key too.
        application_id = str(record.get("graph_key") or record.get("case_id") or "")
        seeds = self._graph_index.application_seed_ids(application_id, self._context_hops)
        task = self._build_task(application_id, catalog, ocel_text, seeds)
        call: dict = {
            "pass": "predict-agentic" if self._use_tools else "predict-graph-single-shot",
            "ok": True,
            "cached": False,
            "system": self._system_prompt,
            "user": task,
            "assistant": None,
            "transcript": [],
            "error": None,
            "max_tool_iterations": self._max_tool_iterations,
            "reasoning_effort": self._reasoning_effort,
            "tools_called": [],
            "retried_for_tool_use": False,
        }

        result, last_exc = await self._run_agent(task, application_id)
        if result is None:
            call["ok"] = False
            call["error"] = str(last_exc)
            return [], 0.0, f"error: {last_exc}", call

        messages = list(getattr(result, "messages", []) or [])
        tools_called = self._tools_called(messages)

        # The whole point of the graph arm is the comparative lookup; if the
        # agent skipped it, re-ask once with the protocol spelled out.
        has_seeds = bool(seeds.get("decisions") or seeds.get("rationales"))
        if (
            self._require_similarity_lookup
            and has_seeds
            and not (set(tools_called) & _SIMILARITY_TOOLS)
        ):
            nudged = f"{task}\n\n{_TOOL_NUDGE}"
            retry_result, _ = await self._run_agent(nudged, application_id)
            if retry_result is not None:
                result = retry_result
                messages = list(getattr(result, "messages", []) or [])
                tools_called = self._tools_called(messages)
                call["user"] = nudged
                call["retried_for_tool_use"] = True

        call["tools_called"] = tools_called
        call["transcript"] = [
            {
                "source": getattr(m, "source", "?"),
                "type": type(m).__name__,
                "content": _render_content(getattr(m, "content", "")),
            }
            for m in messages
        ]
        final_text = self._final_text(messages)
        call["assistant"] = final_text

        parsed = _extract_json_object(final_text)
        if parsed is None:
            call["ok"] = False
            call["error"] = "agent did not return a parseable JSON answer"
            return [], 0.0, "error: no JSON answer", call

        activities = self._normalize(parsed.get("activities", []), allowed)
        confidence = self._as_float(parsed.get("confidence", 0.0))
        reasoning = str(parsed.get("reasoning", "")).strip()
        return activities, confidence, reasoning, call

    async def _run_agent(self, task: str, current_app: str | None = None):
        """Run a fresh agent on *task*, retrying transient 429s.

        Returns ``(result, last_exception)``; ``result`` is None when every
        attempt failed.
        """
        last_exc: BaseException | None = None
        for attempt in range(_RATE_LIMIT_RETRIES + 1):
            try:
                agent = self._build_agent(current_app)
                return await agent.run(task=task), None
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if _is_rate_limit_error(exc) and attempt < _RATE_LIMIT_RETRIES:
                    delay = min(
                        _RATE_LIMIT_BASE_SLEEP * (2 ** attempt), _RATE_LIMIT_MAX_SLEEP
                    ) + random.uniform(0, 1)
                    logger.warning(
                        "agentic reconstruction rate-limited (429); backing off "
                        "%.1fs (attempt %d/%d)", delay, attempt + 1,
                        _RATE_LIMIT_RETRIES,
                    )
                    await asyncio.sleep(delay)
                    continue
                break
        return None, last_exc

    @staticmethod
    def _tools_called(messages: list) -> list[str]:
        """Names of the tools the agent actually invoked, in call order."""
        names: list[str] = []
        for msg in messages:
            content = getattr(msg, "content", None)
            if not isinstance(content, list):
                continue
            for item in content:
                name = getattr(item, "name", None)
                # A request carries arguments; the execution result does not.
                if name is not None and getattr(item, "arguments", None) is not None:
                    names.append(str(name))
        return names

    @staticmethod
    def _final_text(messages: list) -> str:
        for msg in reversed(messages):
            if getattr(msg, "source", None) == "user":
                continue
            content = getattr(msg, "content", None)
            if isinstance(content, str) and content.strip():
                return content
        return ""

    @staticmethod
    def _normalize(raw, allowed: dict) -> list[str]:
        if isinstance(raw, str):
            raw = [raw]
        out: list[str] = []
        for value in raw or []:
            canonical = allowed.get(str(value).strip().lower(), "")
            if canonical:  # keep order AND duplicates
                out.append(canonical)
        return out

    @staticmethod
    def _as_float(value) -> float:
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0

    async def close(self) -> None:
        close = getattr(self._client, "close", None)
        if close is None:
            return
        result = close()
        if inspect.isawaitable(result):
            await result
