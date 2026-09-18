"""The four retrieval tools the reconstruction agent can call.

Each tool is a plain, typed, documented callable (AutoGen builds its JSON schema
from the signature + docstring). They close over a shared read-only
:class:`~src.graph_index.GraphIndex` and :class:`~src.ocel_lookup.OcelStore` and
always return a string, so the agent can chain them:

    find_similar_decisions / find_similar_rationals  -> hybrid (semantic +
        structural) nearest nodes, each with its hops-neighborhood, tagged with
        the application they belong to.
    get_ocel_of_application            -> that application's corrupted OCEL.
    get_application_context_subgraph   -> the hops-neighborhood around an
        application's ActivityCluster.
"""

from __future__ import annotations

from typing import Callable

from src.graph_index import GraphIndex
from src.ocel_lookup import OcelStore


def build_tools(
    graph_index: GraphIndex,
    ocel_store: OcelStore,
    *,
    default_k: int = 5,
    default_hops: int = 1,
    current_app: str | None = None,
) -> list[Callable[..., str]]:
    """Return the four tool callables bound to *graph_index* / *ocel_store*.

    Application ids reach the agent as-is. They used to be rewritten to opaque
    tokens here, because the ids themselves spelled out the pattern size and the
    hidden run's position; those ids are now opaque where they are minted, so
    there is nothing left to rewrite. What an id still reveals is which base case
    it belongs to, and that is load-bearing rather than leaked: the sibling guard
    below needs it to recognise another sampling of the current case.

    *current_app* is the application under reconstruction. Its OCEL timeline and
    its subgraph are already quoted in the task, so fetching either of them again
    only burns the agent's tool budget: both lookups refuse the current
    application and redirect to the similarity tools instead. When the index has
    ``exclude_same_case`` set they also refuse any OTHER sampling of that case,
    which is a correctness guard rather than a budget one -- a sibling hides a
    different email, so its log still shows this case's answer.
    """

    def _is_current(app_id: str) -> bool:
        return bool(current_app) and str(app_id) == current_app

    def _is_sibling_occurrence(app_id: str) -> bool:
        """Another sampling of the case under reconstruction.

        The corpus holds few base applications sampled many times, each hiding a
        DIFFERENT email. A sibling's observed timeline therefore still contains
        the activities this case must predict, so reading it is reading the
        answer. Excluding siblings from the similarity ranking is not enough on
        its own: these tools take an application id as an argument, and the
        agent can pick one up anywhere a ProcessInstance is rendered.

        Follows the index's own ``exclude_same_case`` setting so a run that
        deliberately permits same-case retrieval stays internally consistent.
        """
        if not (current_app and graph_index.exclude_same_case):
            return False
        return graph_index.is_same_base_case(str(app_id), current_app)

    def _redirect(tool: str, what: str) -> str:
        return (
            f"error in {tool}: refused - this is the CURRENT application and its "
            f"{what} is already quoted verbatim in your task, so this call adds "
            "nothing. Call find_similar_decisions or find_similar_rationals on a "
            "seed id from the task, then run this tool on a COMPARABLE "
            "application from those results."
        )

    def _refuse_sibling(tool: str) -> str:
        return (
            f"error in {tool}: refused - that application is another sampling of "
            "the SAME case you are reconstructing, differing only in which email "
            "is hidden. Its log therefore shows the activities you are being "
            "asked to predict, so it is not evidence. Pick a hit from a "
            "genuinely different application instead."
        )

    def find_similar_decisions(
        decision_node_id: str,
        k: int = default_k,
        hops: int = default_hops,
    ) -> str:
        """Find decisions most similar to a given decision node.

        Ranks other DecisionNodes by a HYBRID score that weights semantic
        (text embedding) and structural (FastRP embedding) cosine similarity,
        and attaches each hit's local neighborhood so you can see its rationale,
        evidence and outcome.

        Args:
            decision_node_id: Stable DecisionNode id (e.g. "dec_1a2b..."), taken
                from the current application's subgraph.
            k: How many similar decisions to return (default from config).
            hops: Neighborhood radius (in graph hops) returned around each hit.

        Returns:
            A human-readable ranked list; each hit shows its id, score
            breakdown, owning application id, text and neighborhood subgraph.
        """
        try:
            return graph_index.similar_decisions_text(
                decision_node_id, k, hops, current_app=current_app,
            )
        except Exception as exc:  # noqa: BLE001 - surface as a tool result
            return f"error in find_similar_decisions: {exc}"

    def find_similar_rationals(
        rationale_id: str,
        k: int = default_k,
        hops: int = default_hops,
    ) -> str:
        """Find rationales most similar to a given rationale node.

        Ranks other RationaleNodes by the same HYBRID (semantic + structural)
        score and attaches each hit's local neighborhood (the decision it
        justifies, supporting evidence, etc.).

        Args:
            rationale_id: Stable RationaleNode id (e.g. "rat_9f8e..."), taken
                from the current application's subgraph.
            k: How many similar rationales to return (default from config).
            hops: Neighborhood radius (in graph hops) returned around each hit.

        Returns:
            A human-readable ranked list; each hit shows its id, score
            breakdown, owning application id, text and neighborhood subgraph.
        """
        try:
            return graph_index.similar_rationales_text(
                rationale_id, k, hops, current_app=current_app,
            )
        except Exception as exc:  # noqa: BLE001 - surface as a tool result
            return f"error in find_similar_rationals: {exc}"

    def get_ocel_of_application(application_id: str) -> str:
        """Return an application's observed OCEL timeline (its real event log).

        The timeline is the observed (corrupted) case: that case's hidden
        activities sit behind a single internal_email_sent event. The
        ground-truth answer attributes are always removed; the email
        Subject/Body are shown only when the email is in scope for this run.

        Call this on a COMPARABLE application returned by the similarity tools,
        to learn how an observed log names the steps of a case in this situation.
        Two ids are rejected: the CURRENT application, whose timeline is already
        in the task, and any other sampling of that same case, whose log would
        show the activities you are being asked to predict.

        Args:
            application_id: Application id of another case, as reported by the
                similarity tools, e.g.
                "application_652823628__3f9c2d18b04a". Never the current
                application's id.

        Returns:
            A chronological, sanitised rendering of that application's events.
        """
        if _is_current(application_id):
            return _redirect("get_ocel_of_application", "observed OCEL timeline")
        if _is_sibling_occurrence(application_id):
            return _refuse_sibling("get_ocel_of_application")
        try:
            return ocel_store.get_ocel_text(application_id)
        except Exception as exc:  # noqa: BLE001 - surface as a tool result
            return f"error in get_ocel_of_application: {exc}"

    def get_application_context_subgraph(
        application_id: str,
        hops: int = default_hops,
    ) -> str:
        """Return the context subgraph around an application's ActivityCluster.

        Actor nodes are shown but never expanded through (they would glue
        unrelated applications together); raw OCEL Event nodes are dropped when
        the run filters them.

        Call this on a COMPARABLE application to read its evidence spans beside
        its log. Two ids are rejected: the CURRENT application, whose subgraph is
        already in the task, and any other sampling of that same case.

        Args:
            application_id: Application/case id of another case whose
                ActivityCluster anchors the subgraph. Never the current
                application's id.
            hops: Neighborhood radius (in graph hops) to expand around it.

        Returns:
            The subgraph as node lines (with ids you can feed to the similarity
            tools) and relationship lines.
        """
        if _is_current(application_id):
            return _redirect("get_application_context_subgraph", "subgraph")
        if _is_sibling_occurrence(application_id):
            return _refuse_sibling("get_application_context_subgraph")
        try:
            return graph_index.application_subgraph_text(application_id, hops)
        except Exception as exc:  # noqa: BLE001 - surface as a tool result
            return f"error in get_application_context_subgraph: {exc}"

    return [
        find_similar_decisions,
        find_similar_rationals,
        get_ocel_of_application,
        get_application_context_subgraph,
    ]
