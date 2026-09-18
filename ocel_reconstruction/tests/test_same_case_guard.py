"""A case must never retrieve another occurrence of itself.

The corpus holds only ~20 base applications, each sampled many times at
different absorption points. Two occurrences of one application share almost
everything except WHICH email is hidden, so an occurrence's observed OCEL
timeline contains the activities another occurrence is being asked to predict.
``GraphIndex`` guards against that twice: it always drops candidates from the
query's own application, and ``exclude_same_case`` additionally drops every
other occurrence of the same base case.

Both guards read ``_app_of_node``, and both silently became no-ops when the
case's work moved onto the ActivityCluster: the lookup walked with
``_neighborhood``, which stops at boundary labels, and the cluster is one -- so
no Decision/Rationale could reach its ProcessInstance any more. Nothing failed;
the tools just started reporting ``application=unknown`` and matching cases
against themselves. These tests pin the resolution and both guards.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_RECON = Path(__file__).resolve().parents[1]
if str(_RECON) not in sys.path:
    sys.path.insert(0, str(_RECON))

from src.graph_index import GraphIndex  # noqa: E402


def _node(nid: str, label: str, props: dict) -> dict:
    return {"type": "node", "id": nid, "labels": [label], "properties": props}


def _rel(start: str, rel: str, end: str) -> dict:
    return {"type": "relationship", "label": rel, "start": {"id": start},
            "end": {"id": end}}


def _occurrence(graph: list, n: int, app_id: str, vec: list[float]) -> None:
    """One application: PI <- cluster <- decision/rationale, as ingest builds it."""
    pi, cl, dec, rat = f"pi{n}", f"cl{n}", f"dec{n}", f"rat{n}"
    embed = {"text_embedding": vec, "fastrp_embedding": vec}
    graph += [
        _node(pi, "ProcessInstance", {"id": app_id, "external_system_id": app_id,
                                      "name": "Bank loan application"}),
        _node(cl, "ActivityCluster", {"id": f"cluster_{app_id}"}),
        _node(dec, "DecisionNode",
              {"id": f"dec_{n}", "text": "offer returned, checking documents", **embed}),
        _node(rat, "RationaleNode",
              {"id": f"rat_{n}", "text": "the returned offer is being checked", **embed}),
        _rel(cl, "part_of", pi),
        _rel(cl, "has_decision", dec),
        _rel(dec, "justified_by", rat),
    ]


@pytest.fixture
def graph_path(tmp_path):
    """Three occurrences: two of base application_1, one of application_2.

    The two application_1 occurrences get near-identical embeddings, so the
    sibling occurrence is the TOP-ranked neighbour. That is the realistic
    shape -- two samples of one case look alike -- and it means a guard that
    does nothing cannot accidentally pass.
    """
    graph: list = []
    _occurrence(graph, 1, "application_1__3f9c2d18b04a", [1.0, 0.0, 0.0])
    _occurrence(graph, 2, "application_1__b71e0a4c9d52", [0.99, 0.1, 0.0])
    _occurrence(graph, 3, "application_2__c04d7f1a6e83", [0.6, 0.8, 0.0])
    p = tmp_path / "graph.json"
    p.write_text("\n".join(json.dumps(r) for r in graph), encoding="utf-8")
    return p


def test_decisions_resolve_to_their_application_through_the_cluster(graph_path):
    # The regression itself: with the cluster in the path this map went empty,
    # which is what made both guards silently stop working.
    idx = GraphIndex(graph_path)
    assert idx._app_of_node, "no Decision/Rationale resolved to an application"
    assert set(idx._app_of_node.values()) == {
        "application_1__3f9c2d18b04a",
        "application_1__b71e0a4c9d52",
        "application_2__c04d7f1a6e83",
    }
    assert set(idx._base_of_node.values()) == {"application_1", "application_2"}


def test_similar_decisions_never_report_an_unknown_application(graph_path):
    out = GraphIndex(graph_path).similar_decisions_text("dec_1", k=5, hops=2)
    assert "application=unknown" not in out


def test_own_application_is_always_excluded(graph_path):
    # Guard 1, unconditional: a case's own decisions are already given as seeds.
    out = GraphIndex(graph_path).similar_decisions_text("dec_1", k=5, hops=2)
    assert "application_1__3f9c2d18b04a" not in out


def test_exclude_same_case_drops_other_occurrences_of_the_same_base(graph_path):
    off = GraphIndex(graph_path, exclude_same_case=False)
    on = GraphIndex(graph_path, exclude_same_case=True)
    other_occurrence = "application_1__b71e0a4c9d52"
    unrelated = "application_2__c04d7f1a6e83"

    # Off, the sibling occurrence is reachable -- that is the leak.
    assert other_occurrence in off.similar_decisions_text("dec_1", k=5, hops=2)
    # On, it is gone and only the genuinely different application remains.
    on_text = on.similar_decisions_text("dec_1", k=5, hops=2)
    assert other_occurrence not in on_text
    assert unrelated in on_text


def test_exclude_same_case_also_applies_to_rationales(graph_path):
    on = GraphIndex(graph_path, exclude_same_case=True)
    assert "application_1__b71e0a4c9d52" not in on.similar_rationales_text(
        "rat_1", k=5, hops=2,
    )


def test_chaining_off_a_foreign_node_cannot_return_the_case_being_solved(graph_path):
    """The guard must follow the RECONSTRUCTION target, not the query node.

    The agent does not only query its own seeds; it takes a hit from another
    case and queries that. Excluding just the query's application then excludes
    the wrong case and offers the reconstruction target -- and its siblings --
    straight back. Every sibling that survived the same-case filter in a real
    run arrived this way: 16 hits, all from chained foreign queries, none from
    a seed.

    Here application_2 is the foreign node the agent chained to, while
    application_1 is still the case under reconstruction.
    """
    idx = GraphIndex(graph_path, exclude_same_case=True)
    foreign = "dec_3"                     # belongs to application_2

    unanchored = idx.similar_decisions_text(foreign, k=5, hops=2)
    assert "application_1__3f9c2d18b04a" in unanchored, (
        "fixture must actually surface the target when unanchored"
    )

    anchored = idx.similar_decisions_text(
        foreign, k=5, hops=2, current_app="application_1__3f9c2d18b04a",
    )
    assert "application_1__3f9c2d18b04a" not in anchored  # the case itself
    assert "application_1__b71e0a4c9d52" not in anchored  # its sibling


def test_the_tools_anchor_similarity_to_the_current_application(graph_path):
    tools = _tools(graph_path, "application_1__3f9c2d18b04a")
    out = tools["find_similar_decisions"]("dec_3")   # chain off application_2
    assert "application_1__3f9c2d18b04a" not in out
    assert "application_1__b71e0a4c9d52" not in out


# ── the tools refuse a sibling id however the agent obtained it ─────────────

class _Ocel:
    """Stand-in OcelStore; returns a marker so a leak is unmistakable."""

    def get_ocel_text(self, application_id: str) -> str:
        return f"TIMELINE OF {application_id}"


def _tools(graph_path, current_app, *, exclude_same_case=True):
    from src.tools import build_tools

    idx = GraphIndex(graph_path, exclude_same_case=exclude_same_case)
    return {t.__name__: t for t in build_tools(idx, _Ocel(), current_app=current_app)}


CURRENT = "application_1__3f9c2d18b04a"
SIBLING = "application_1__b71e0a4c9d52"
UNRELATED = "application_2__c04d7f1a6e83"


def test_ocel_lookup_refuses_a_sibling_occurrence(graph_path):
    # Filtering the similarity ranking cannot cover this: the agent passes an id
    # directly, and it can read one off any rendered ProcessInstance.
    out = _tools(graph_path, CURRENT)["get_ocel_of_application"](SIBLING)
    assert "refused" in out and "SAME case" in out
    assert "TIMELINE OF" not in out


def test_subgraph_lookup_refuses_a_sibling_occurrence(graph_path):
    out = _tools(graph_path, CURRENT)["get_application_context_subgraph"](SIBLING)
    assert "refused" in out and "SAME case" in out


def test_a_genuinely_different_application_is_still_allowed(graph_path):
    # The guard must not break the tool's actual purpose.
    out = _tools(graph_path, CURRENT)["get_ocel_of_application"](UNRELATED)
    assert out == f"TIMELINE OF {UNRELATED}"


def test_the_current_application_still_gets_the_budget_redirect(graph_path):
    out = _tools(graph_path, CURRENT)["get_ocel_of_application"](CURRENT)
    assert "CURRENT application" in out


def test_the_tool_guard_follows_the_index_setting(graph_path):
    # With same-case retrieval deliberately permitted the tools must not
    # second-guess it, or the two guards would disagree about one run.
    out = _tools(graph_path, CURRENT, exclude_same_case=False)["get_ocel_of_application"](SIBLING)
    assert out == f"TIMELINE OF {SIBLING}"


# ── the flag has to survive being typed on the command line ────────────────

def test_a_mistyped_boolean_raises_instead_of_becoming_false():
    import run

    for good, expected in (("true", True), ("TRUE", True), ("1", True),
                           ("false", False), ("no", False), (" off ", False)):
        assert run._str2bool(good, flag="--x") is expected

    # The real incident: exclude_same_case was submitted as "ture" and the old
    # membership test quietly made it False, so a whole experiment ran with the
    # same-case guard off and only a log line recorded it.
    for typo in ("ture", "yess", "", "None"):
        with pytest.raises(ValueError, match="not a boolean"):
            run._str2bool(typo, flag="--exclude-same-case")
