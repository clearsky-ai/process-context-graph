"""Renderer tests for the shared subgraph renderer.

``render_island`` is the only thing every prompt's graph text goes through, so
these exercise it directly with stub accessors instead of going via a graph
holder. The fixture mirrors the shape the ingest now produces: an unnamed
``ActivityCluster`` that the work hangs off, ``part_of`` its ``ProcessInstance``,
which carries the case-level attributes as metadata (no value nodes).
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from src.graph_json import _SPAN_INDEX_RE, render_island

_VERBS = ("prepared", "reviewed", "ran", "completed", "approved", "signed_off")


class _StubGraph:
    """The four accessors ``render_island`` asks for, over plain dicts."""

    def __init__(
        self,
        nodes: dict[str, tuple[str, dict]],
        edges: list[tuple[str, str, str]],
    ) -> None:
        self._nodes = nodes
        self._edges = edges

    def _label(self, nid: str) -> str:
        return self._nodes[nid][0]

    def _props(self, nid: str) -> dict:
        return self._nodes[nid][1]

    def _span_index(self, nid: str) -> int:
        pid = str(self._props(nid).get("id") or nid)
        match = _SPAN_INDEX_RE.search(pid)
        return int(match.group(1)) if match else 0

    def _out_edges(self, nid: str) -> list[tuple[str, str]]:
        return [(dst, rel) for src, rel, dst in self._edges if src == nid]

    def render(self, *, anchors=frozenset(), mask_execution_verbs=True) -> str:
        return render_island(
            list(self._nodes),
            primary_label=self._label,
            props_of=self._props,
            span_index_of=self._span_index,
            out_edges_of=self._out_edges,
            anchors=anchors,
            mask_execution_verbs=mask_execution_verbs,
        )

    def relationships(self, **kwargs) -> list[str]:
        """Just the ``Relationships:`` lines, for edge-level assertions."""
        return self.render(**kwargs).split("Relationships:\n", 1)[1].splitlines()


def _case_graph() -> _StubGraph:
    """One case, in the shape the ingest emits."""
    return _StubGraph(
        nodes={
            "e1": ("EvidenceNode", {
                "id": "ev_mail1__s1", "text": "I created the offer.",
            }),
            # An ActivityCluster carries nothing but its id, which _SKIP_KEYS
            # already hides -- that is what makes it render as "no details".
            "ac1": ("ActivityCluster", {"id": "cluster_app1"}),
            "p1": ("ProcessInstance", {"name": "Bank loan application"}),
            "d1": ("DecisionNode", {"name": "Approve"}),
            "a1": ("Actor", {"name": "Ada Lovelace", "actor_type": "Person"}),
        },
        edges=[
            ("ac1", "part_of", "p1"),
            ("a1", "prepared", "ac1"),
            ("a1", "completed", "ac1"),
            ("a1", "produced_evidence", "e1"),
            ("ac1", "produced", "e1"),
            ("ac1", "has_decision", "d1"),
            ("d1", "acted_on", "ac1"),
            ("d1", "supported_by", "e1"),
        ],
    )


def test_node_and_relationship_sections() -> None:
    graph = _StubGraph(
        nodes={
            "e1": ("EvidenceNode", {
                "id": "ev_mail1__s1", "text": "I created the offer.",
            }),
            "d1": ("DecisionNode", {"name": "Approve"}),
        },
        edges=[("d1", "supported_by", "e1")],
    )
    assert graph.render(anchors={"e1"}) == (
        "Node:\n"
        'E1 EvidenceNode("I created the offer.", this email, span=1)\n'
        "D1 DecisionNode(Approve)\n"
        "\n"
        "Relationships:\n"
        "D1 is supported_by E1"
    )


def test_cluster_renders_unnamed_and_before_its_case() -> None:
    body = _case_graph().render()
    # No special-casing: the cluster has no renderable property of its own.
    assert "AC1 ActivityCluster(no details)" in body
    assert "P1 ProcessInstance(Bank loan application)" in body
    lines = body.splitlines()
    assert lines.index("AC1 ActivityCluster(no details)") < lines.index(
        "P1 ProcessInstance(Bank loan application)"
    )


def test_work_hangs_off_the_cluster() -> None:
    rels = _case_graph().relationships()
    assert "AC1 is produced E1" in rels
    assert "AC1 is has_decision D1" in rels
    assert "D1 is acted_on AC1" in rels
    assert "AC1 is part_of P1" in rels


def test_masking_neutralises_every_execution_verb() -> None:
    graph = _StubGraph(
        nodes={
            "ac1": ("ActivityCluster", {"id": "cluster_app1"}),
            **{
                f"a{i}": ("Actor", {"name": f"Actor {i}"})
                for i, _ in enumerate(_VERBS, start=1)
            },
        },
        edges=[(f"a{i}", verb, "ac1") for i, verb in enumerate(_VERBS, start=1)],
    )
    rels = graph.relationships()
    assert all(line.split(" is ")[1].startswith("linked_to") for line in rels)
    assert not any(verb in " ".join(rels) for verb in _VERBS)


def test_masking_collapses_two_verbs_between_the_same_pair() -> None:
    graph = _case_graph()
    # One actor both prepared and completed the case: two lines unmasked...
    assert [r for r in graph.relationships(mask_execution_verbs=False)
            if r.startswith("A1 is")] == [
        "A1 is completed AC1",
        "A1 is prepared AC1",
        "A1 is produced_evidence E1",
    ]
    # ...and one linked_to line plus the untouched produced_evidence when masked.
    assert [r for r in graph.relationships() if r.startswith("A1 is")] == [
        "A1 is linked_to AC1",
        "A1 is produced_evidence E1",
    ]


def test_masking_leaves_every_other_edge_verbatim() -> None:
    masked = _case_graph().relationships()
    plain = _case_graph().relationships(mask_execution_verbs=False)
    untouched = {
        r for r in plain
        if not any(f" is {verb} " in f" {r} " for verb in _VERBS)
    }
    assert untouched <= set(masked)
