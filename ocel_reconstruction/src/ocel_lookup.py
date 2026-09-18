"""Per-application corrupted OCEL lookup for the agentic reconstruction.

The OCEL Email Generator writes, for every occurrence, a
``<graph_key>/filled_ocel.json`` next to ``generated.jsonl`` -- the CORRUPTED
OCEL 2.0 log for that occurrence with the hidden span replaced by a single
``internal_email_sent`` event whose ``Subject`` / ``Body`` were filled in by the
generator. That event also carries ``true_members`` and ``original_event_ids``,
which are the GROUND-TRUTH answer for the reconstruction task.

``OcelStore`` maps an ``application_id`` (the corrupted case id, e.g.
``Application_652823628__3f9c2d18b04a``, which is ``case_id`` in
``generated.jsonl`` and ``ProcessInstance.external_system_id`` in the graph) to
its ``filled_ocel.json`` and renders it for the ``get_ocel_of_application`` tool
after sanitising it:

* ALWAYS strip ``true_members`` / ``original_event_ids`` (never leak the answer).
* When ``use_email`` is False, also redact the email ``Subject`` / ``Body`` so the
  no-email arms cannot recover the email text through this tool.
"""

from __future__ import annotations

import json
from pathlib import Path

# Email-event attributes that MUST never reach the model (the ground truth).
_ANSWER_ATTRS = {"true_members", "original_event_ids"}
# Email-event attributes redacted when use_email is False.
_MAIL_ATTRS = {"Subject", "Body"}
_REDACTED = "[redacted]"


def _attr_value(attrs: list, name: str):
    for a in attrs or []:
        if isinstance(a, dict) and a.get("name") == name:
            return a.get("value")
    return None


class OcelStore:
    """Resolve application ids to their sanitised, rendered corrupted OCEL."""

    def __init__(
        self,
        emails_root: "str | Path",
        records: list[dict],
        *,
        use_email: bool,
        email_activity: str = "internal_email_sent",
    ) -> None:
        self._emails_root = Path(emails_root)
        self._use_email = bool(use_email)
        self._email_activity = email_activity
        # application id (case_id) -> graph_key, plus the set of known graph_keys
        self._gk_by_app: dict[str, str] = {}
        self._graph_keys: set[str] = set()
        for rec in records:
            gk = rec.get("graph_key")
            if not gk:
                continue
            self._graph_keys.add(str(gk))
            case_id = rec.get("case_id")
            if case_id:
                self._gk_by_app.setdefault(str(case_id), str(gk))

    # ------------------------------------------------------------------ #
    def _resolve_graph_key(self, application_id: str) -> "str | None":
        application_id = str(application_id)
        if application_id in self._gk_by_app:
            return self._gk_by_app[application_id]
        if application_id in self._graph_keys:  # agent passed a graph_key directly
            return application_id
        return None

    def _load_ocel(self, graph_key: str) -> "dict | None":
        path = self._emails_root / graph_key / "filled_ocel.json"
        if not path.is_file():
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def get_ocel_text(self, application_id: str) -> str:
        """Return the sanitised, rendered corrupted OCEL for *application_id*."""
        graph_key = self._resolve_graph_key(application_id)
        if graph_key is None:
            return (
                f"No OCEL found for application {application_id!r}. Use an "
                "application id reported by the similarity tools or the current "
                "application's id."
            )
        ocel = self._load_ocel(graph_key)
        if ocel is None:
            return f"filled_ocel.json is missing for application {application_id!r}."
        return self._render(application_id, ocel)

    # ------------------------------------------------------------------ #
    def _render(self, application_id: str, ocel: dict) -> str:
        events = ocel.get("events") or []
        rows: list[tuple[str, str]] = []  # (time, rendered line)
        for ev in events:
            etype = ev.get("type", "?")
            time = str(ev.get("time", ""))
            rels = ev.get("relationships") or []
            obj_ids = [r.get("objectId") for r in rels if isinstance(r, dict)]
            obj_ids = [o for o in obj_ids if o]
            if etype == self._email_activity:
                rows.append((time, self._render_email_event(ev, obj_ids)))
            else:
                objs = f" [objects: {', '.join(obj_ids)}]" if obj_ids else ""
                rows.append((time, f"- {time} | {etype}{objs}"))
        rows.sort(key=lambda r: r[0])
        header = (
            f"Corrupted/observed OCEL for application {application_id} "
            "(the hidden activities are behind the internal_email_sent event; "
            "ground-truth answer attributes are removed):"
        )
        body = "\n".join(line for _t, line in rows) or "(no events)"
        return f"{header}\nEvents (chronological):\n{body}"

    def _render_email_event(self, ev: dict, obj_ids: list[str]) -> str:
        attrs = ev.get("attributes") or []
        time = str(ev.get("time", ""))
        email_id = _attr_value(attrs, "EmailID")
        sender = _attr_value(attrs, "Sender")
        subject = _attr_value(attrs, "Subject")
        body = _attr_value(attrs, "Body")
        if not self._use_email:
            subject = _REDACTED
            body = _REDACTED
        parts = [f"- {time} | {self._email_activity} (HIDDEN activities are here)"]
        meta = []
        if email_id:
            meta.append(f"EmailID={email_id}")
        if sender:
            meta.append(f"Sender={sender}")
        if obj_ids:
            meta.append(f"objects: {', '.join(obj_ids)}")
        if meta:
            parts.append("  " + " | ".join(meta))
        parts.append(f'  Subject: {subject if subject is not None else ""}')
        parts.append(f'  Body: {body if body is not None else ""}')
        return "\n".join(parts)
