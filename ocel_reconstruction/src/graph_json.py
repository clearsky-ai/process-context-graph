"""How a context-graph subgraph is rendered for the model.

The one place that decides what a subgraph looks like in a prompt: ``Node:``
lines ``TAG Label(details)`` followed by ``Relationships:`` lines
``SRC_TAG is REL DST_TAG``. :class:`~src.graph_index.GraphIndex` selects WHICH
nodes to show (the hops-bounded neighborhood around an application) and calls
:func:`render_island` to turn them into that text, so the task subgraph and
every graph-returning tool share one shape.

The graph already has the shape the prompt wants: the ingest seeds one unnamed
``ActivityCluster`` per ``ProcessInstance``, the case's work hangs off the
cluster, and the case-level attributes stay on the ProcessInstance. So this
module renders what it is given rather than reshaping anything -- an
``ActivityCluster`` shows as ``AC1 ActivityCluster(no details)`` simply because
the node carries no renderable property.

The one transform left is ``mask_execution_verbs`` (default on): the six Actor
execution verbs pointing at a cluster collapse to a single neutral
``linked_to``. Each of those verbs is an LLM's one-word summary of what an actor
did, so ``approved`` or ``reviewed`` hints at the kind of activity behind it;
masking keeps the actor attached to the case without that hint. Every other edge
renders verbatim.
"""

from __future__ import annotations

import re

# Body-span EvidenceNode ``id`` properties end in ``__s<index>`` (1-based
# reading order). Matched against the ``id`` PROPERTY, not the APOC node id.
_SPAN_INDEX_RE = re.compile(r"__s(\d+)$")

# Max characters kept when rendering node text. A single EvidenceNode can hold a
# whole email body (span extraction sometimes emits one span for the whole mail),
# so this must be large enough not to chop off later activities. Only a very
# large safety guard against pathological blobs.
_MAX_TEXT_CHARS = 4000

# Generic helper labels the KG builder stamps on nodes; they carry no meaning and
# can shadow the real label, so we strip them before consuming the graph (the
# ingest component already flushes these, but we do it again defensively).
_HELPER_LABELS = {"__Entity__", "__KGBuilder__", "inline_text"}

# Bookkeeping keys with no classification signal.
_SKIP_KEYS = {
    "id", "external_system_id", "ocel_object_id", "ocel_event_id",
    "raw_data_ref", "system_id", "source_id", "recorded_at",
    "last_processed_at", "signal_source", "valid_from", "valid_to",
    "started_at", "decision_time",
}
_PREF_ORDER = [
    "name", "final_choice", "decision_type", "outcome_status",
    "decision_status", "frequency", "actor_type", "confidence", "text",
]

_EVIDENCE = "EvidenceNode"
_ACTOR = "Actor"
_DECISION = "DecisionNode"
_RATIONALE = "RationaleNode"
_EVENT = "Event"
_ACTIVITY = "Activity"
_PROCESS = "ProcessInstance"
_LEAKY_LABELS = {_EVENT, _ACTIVITY}

# The case's work node, seeded by the ingest: one unnamed cluster per
# ProcessInstance, ``part_of`` it. Distinct from _ACTIVITY, which the ingest no
# longer emits at all.
_CLUSTER = "ActivityCluster"

# The Actor execution verbs. Each is an LLM's one-word summary of what an actor
# did on the case, so it hints at the kind of activity behind it. Masking them
# to _NEUTRAL_REL keeps the actor attached without that hint.
_MASKED_RELS = frozenset({
    "prepared", "reviewed", "ran", "completed", "approved", "signed_off",
})
_NEUTRAL_REL = "linked_to"

_TAG_PREFIX = {
    _EVIDENCE: "E", _ACTOR: "A", _DECISION: "D", _RATIONALE: "R",
    _EVENT: "EV", _ACTIVITY: "ACT", _PROCESS: "P", _CLUSTER: "AC",
}
# Only the relative order matters. The cluster sorts before its ProcessInstance
# so a subgraph reads work-first, then the case it belongs to.
_LABEL_ORDER = {
    _EVIDENCE: 0, _CLUSTER: 1, _PROCESS: 2, _ACTIVITY: 3,
    _DECISION: 4, _RATIONALE: 5, _EVENT: 6, _ACTOR: 7,
}
_LABEL_ORDER_DEFAULT = 8

def _scalar(value):
    """Unwrap an ``apoc`` typed value ``{"type":..,"value":..}`` to its scalar."""
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def _real_labels(labels) -> list[str]:
    """Return the node's labels with generic helper labels removed."""
    return [lab for lab in (labels or []) if lab not in _HELPER_LABELS]


def _clip(text: str) -> str:
    if len(text) > _MAX_TEXT_CHARS:
        return text[:_MAX_TEXT_CHARS] + "\u2026"
    return text


def _format_props(props: dict) -> str:
    parts: list[str] = []
    for key in _PREF_ORDER:
        if key in _SKIP_KEYS:
            continue
        value = _scalar(props.get(key))
        if value in (None, ""):
            continue
        text = _clip(str(value))
        if key == "name":
            parts.append(text)
        elif key == "text":
            parts.append(f'"{text}"')
        else:
            parts.append(f"{key}={text}")
    return " | ".join(parts) if parts else "no details"


def _node_text(props: dict) -> str:
    """Quoted-content field: ``text``, falling back to ``name``."""
    text = _scalar(props.get("text")) or _scalar(props.get("name")) or ""
    return _clip(str(text).strip()) or "(no text)"


def _evidence_details(props: dict, is_anchor: bool, span_idx: int) -> str:
    origin = "this email" if is_anchor else "other"
    span_bit = f", span={span_idx}" if span_idx else ""
    return f'"{_node_text(props)}", {origin}{span_bit}'


def _rationale_details(props: dict) -> str:
    return f'"{_node_text(props)}"'


def render_island(
    node_ids,
    *,
    primary_label,
    props_of,
    span_index_of,
    out_edges_of,
    anchors=frozenset(),
    mask_execution_verbs=True,
) -> str:
    """Render *node_ids* as ``Node:`` / ``Relationships:`` sections.

    The caller supplies small accessors so this works for any graph holder:

    * ``primary_label(nid) -> str``
    * ``props_of(nid) -> dict``            (raw properties; apoc-typed ok)
    * ``span_index_of(nid) -> int``        (0 when the node has no span order)
    * ``out_edges_of(nid) -> [(dst, rel)]`` (outgoing only)
    * ``anchors``                          nodes belonging to THIS email; an
      EvidenceNode outside it renders as ``other``.
    * ``mask_execution_verbs``             collapse the six Actor execution
      verbs pointing at an ActivityCluster to a single neutral ``linked_to``.
    """
    ids = list(node_ids)
    if not ids:
        return ""
    in_island = set(ids)

    def sort_key(nid):
        label = primary_label(nid)
        span_pos = span_index_of(nid) if label == _EVIDENCE else 0
        return (
            _LABEL_ORDER.get(label, _LABEL_ORDER_DEFAULT),
            0 if nid in anchors else 1,
            span_pos,
            str(nid),
        )

    ordered = sorted(ids, key=sort_key)

    tag_of: dict = {}
    counters: dict[str, int] = {}
    for nid in ordered:
        prefix = _TAG_PREFIX.get(primary_label(nid), "N")
        counters[prefix] = counters.get(prefix, 0) + 1
        tag_of[nid] = f"{prefix}{counters[prefix]}"

    node_lines = []
    for nid in ordered:
        label = primary_label(nid)
        props = props_of(nid) or {}
        if label == _EVIDENCE:
            details = _evidence_details(props, nid in anchors, span_index_of(nid))
        elif label == _RATIONALE:
            details = _rationale_details(props)
        else:
            # An ActivityCluster needs no special case: it carries no renderable
            # property, so _format_props yields "no details" on its own.
            details = _format_props(props)
        node_lines.append(f"{tag_of[nid]} {label}({details})")

    seen: set[tuple] = set()
    rel_keys: list[tuple[str, str, str]] = []
    for nid in ids:
        for dst, rel in out_edges_of(nid):
            if dst not in in_island:
                continue
            if (
                mask_execution_verbs
                and rel in _MASKED_RELS
                and _CLUSTER in (primary_label(nid), primary_label(dst))
            ):
                rel = _NEUTRAL_REL
            # Dedupe on the RENDERED rel: an actor who both prepared and
            # completed the same case yields one ``linked_to`` line, not two
            # identical ones.
            key = (nid, rel, dst)
            if key in seen:
                continue
            seen.add(key)
            rel_keys.append((tag_of[nid], rel, tag_of[dst]))

    rel_lines = [f"{src} is {rel} {dst}" for src, rel, dst in sorted(rel_keys)]

    return "\n".join(["Node:", *node_lines, "", "Relationships:", *rel_lines])
