"""In-memory index over a dumped ``graph.json`` for the agentic reconstruction.

Context Graph Ingest exports the whole case graph with
``apoc.export.json.all(..., {useTypes:true})`` as JSONL (one node or
relationship per line), with two vector properties written onto every
``DecisionNode`` / ``RationaleNode`` by ``scripts/embed_graph.py``:

* ``text_embedding``   -- semantic (Azure OpenAI) embedding of the node text.
* ``fastrp_embedding`` -- structural (Neo4j GDS FastRP) embedding.

``GraphIndex`` loads that dump once (no live Cypher) and serves the four
retrieval tools the agent calls:

* similar decisions / rationales -- a HYBRID, weighted mix of semantic and
  structural cosine similarity, with each hit's ``hops``-neighborhood attached
  as context.
* an application's context subgraph -- the ``hops``-neighborhood around the
  ``ProcessInstance`` whose id matches an application id.

Design choices:
  * ``Actor`` nodes are rendered when reached but never traversed THROUGH
    (globally-merged Actors would glue unrelated applications/emails together),
  * ``Event`` / ``Activity`` nodes are dropped from rendered neighborhoods
    unless ``include_event`` is True (they leak the true activity),
  * APOC typed values ``{"type":..,"value":..}`` are unwrapped with ``_scalar``.

Selection happens here; how the selected nodes are turned into prompt text lives
in :mod:`src.graph_json`, including the ``mask_execution_verbs`` setting this
class forwards to it.

Missing-embedding policy is FAIL-HARD: if any ``DecisionNode`` / ``RationaleNode``
lacks ``text_embedding`` or ``fastrp_embedding``, construction raises a
``ValueError`` naming the offending node ids and the missing property, so a
broken/partial ingest surfaces immediately instead of being silently masked.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.graph_json import (
    _CLUSTER,
    _HELPER_LABELS,
    _SPAN_INDEX_RE,
    _real_labels,
    _scalar,
    render_island,
)

# Vector property names written by context_graph_ingest/cga/scripts/embed_graph.py.
SEMANTIC_EMBED_PROPERTY = "text_embedding"
FASTRP_EMBED_PROPERTY = "fastrp_embedding"

_EVIDENCE = "EvidenceNode"
_ACTOR = "Actor"
_DECISION = "DecisionNode"
_RATIONALE = "RationaleNode"
_EVENT = "Event"
_ACTIVITY = "Activity"
_PROCESS = "ProcessInstance"
_LEAKY_LABELS = {_EVENT, _ACTIVITY}
# Reached-but-never-traversed-THROUGH: a different ProcessInstance (or its
# ActivityCluster) is a different application, and an Actor is merged globally
# across cases. Expanding through any of them would merge unrelated apps into
# one "island". Seeds always expand, so an application's OWN cluster is still
# walked through -- only a foreign one reached later is terminal.
_BOUNDARY_LABELS = {_PROCESS, _CLUSTER, _ACTOR}
# A ProcessInstance is only rendered when one of these node types in the SAME
# rendered set connects to it directly; otherwise it is dropped. General rule
# applied by _render_subgraph, so it holds for every graph-returning tool.
# The ActivityCluster belongs here because the case's work now hangs off IT:
# a PI's only direct neighbours are its cluster and the value nodes, so without
# it every application would read as bare and be discarded.
_PI_LINK_LABELS = {_EVIDENCE, _DECISION, _RATIONALE, _CLUSTER}

# Property keys that carry no classification signal when rendering a node, plus
# the two embeddings (huge vectors we never want to print).
_SKIP_KEYS = {
    "id", "external_system_id", "ocel_object_id", "ocel_event_id",
    "raw_data_ref", "system_id", "source_id", "recorded_at",
    "last_processed_at", "valid_from", "valid_to", "superseded_at",
    "started_at", "decision_time", SEMANTIC_EMBED_PROPERTY, FASTRP_EMBED_PROPERTY,
}
# Preferred order for the human-readable detail string of a node.
_PREF_ORDER = [
    "name", "final_choice", "decision_type", "outcome_status",
    "decision_status", "signal_source", "actor_type", "confidence", "text",
]
# Keys tried (in order) to resolve a ProcessInstance's application id.
_APP_ID_KEYS = ("external_system_id", "ocel_object_id", "name", "id")

_MAX_TEXT_CHARS = 2000


def _clip(text: str, limit: int = _MAX_TEXT_CHARS) -> str:
    text = str(text)
    return text if len(text) <= limit else text[:limit] + "\u2026"


def _as_vector(value) -> "list[float] | None":
    """Unwrap an APOC-typed embedding value into a plain list of floats."""
    value = _scalar(value)
    if not isinstance(value, (list, tuple)) or not value:
        return None
    try:
        return [float(_scalar(x)) for x in value]
    except (TypeError, ValueError):
        return None


class GraphIndex:
    """Load ``graph.json`` once and answer similarity / subgraph queries."""

    def __init__(
        self,
        graph_path: "str | Path",
        *,
        alpha: float = 0.5,
        filter_event_node: bool = True,
        exclude_same_case: bool = False,
        mask_execution_verbs: bool = True,
    ) -> None:
        self._alpha = float(alpha)
        self._include_event = not bool(filter_event_node)
        # When True, similarity assigns -inf (excludes) to any candidate whose
        # owning application shares the SAME base case as the query node, so a
        # decision/rationale can never match another occurrence of its own case.
        self._exclude_same_case = bool(exclude_same_case)
        # Render-only: collapse the Actor execution verbs to a neutral edge.
        # Selection is unaffected.
        self._mask_execution_verbs = bool(mask_execution_verbs)

        # apoc_id -> {"labels": [...], "properties": {...}, "auth": bool}
        self._nodes: dict[str, dict] = {}
        # apoc_id -> list of (neighbour_apoc_id, rel_type, direction)
        self._adj: dict[str, list[tuple[str, str, str]]] = {}
        # stable properties.id -> apoc_id
        self._apoc_by_prop_id: dict[str, str] = {}
        # application id -> ProcessInstance apoc ids
        self._pis_by_app: dict[str, list[str]] = {}
        # any ProcessInstance id value (external_system_id / ocel_object_id /
        # name / id) -> ProcessInstance apoc id, so a lookup by the record's
        # graph_key resolves regardless of which field the graph stored it in.
        self._pi_by_key: dict[str, str] = {}
        # ProcessInstance apoc id -> its canonical application id
        self._app_of_pi: dict[str, str] = {}
        # Decision/Rationale apoc id -> its owning application (exact occurrence),
        # and its base case (shared by all occurrences of the same original case).
        self._app_of_node: dict[str, str] = {}
        self._base_of_node: dict[str, str] = {}
        # per-label similarity matrices, filled in _build_label_matrices
        self._label_data: dict[str, dict] = {}

        graph_path = Path(graph_path)
        if not graph_path.exists():
            raise FileNotFoundError(f"graph.json not found: {graph_path}")
        self._load(graph_path)
        self._build_pi_index()
        self._build_node_apps()
        self._build_label_matrices()  # fail-hard on missing embeddings

    # ------------------------------------------------------------------ #
    #  Loading                                                            #
    # ------------------------------------------------------------------ #
    def _load(self, graph_path: Path) -> None:
        with open(graph_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip().rstrip(",")
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                kind = obj.get("type")
                if kind == "node":
                    self._add_node(
                        obj.get("id"), obj.get("labels"),
                        obj.get("properties") or {}, authoritative=True,
                    )
                elif kind == "relationship":
                    start = obj.get("start") or {}
                    end = obj.get("end") or {}
                    self._add_node(start.get("id"), start.get("labels"),
                                   start.get("properties") or {}, authoritative=False)
                    self._add_node(end.get("id"), end.get("labels"),
                                   end.get("properties") or {}, authoritative=False)
                    rel = obj.get("label") or obj.get("type") or "REL"
                    if start.get("id") is not None and end.get("id") is not None:
                        self._add_edge(str(start["id"]), str(end["id"]), rel)
        # Build the stable-id -> apoc-id index from authoritative node props.
        for apoc_id, node in self._nodes.items():
            pid = _scalar((node["properties"] or {}).get("id"))
            if pid:
                self._apoc_by_prop_id.setdefault(str(pid), apoc_id)

    def _add_node(self, apoc_id, labels, properties: dict, *, authoritative: bool) -> None:
        if apoc_id is None:
            return
        apoc_id = str(apoc_id)
        node = self._nodes.get(apoc_id)
        if node is None:
            self._nodes[apoc_id] = {
                "labels": _real_labels(labels),
                "properties": properties or {},
                "auth": authoritative,
            }
            return
        # A node line (authoritative) always wins over relationship-endpoint
        # copies, which may omit the embedding vectors.
        if authoritative:
            if labels:
                node["labels"] = _real_labels(labels)
            if properties:
                node["properties"] = properties
            node["auth"] = True
        else:
            if not node["labels"] and labels:
                node["labels"] = _real_labels(labels)
            if not node["properties"] and properties:
                node["properties"] = properties

    def _add_edge(self, start: str, end: str, rel: str) -> None:
        self._adj.setdefault(start, []).append((end, rel, "out"))
        self._adj.setdefault(end, []).append((start, rel, "in"))

    # ------------------------------------------------------------------ #
    #  Label helpers                                                      #
    # ------------------------------------------------------------------ #
    def _labels(self, apoc_id: str) -> list[str]:
        node = self._nodes.get(apoc_id)
        return (node["labels"] if node else []) or []

    def _has_label(self, apoc_id: str, label: str) -> bool:
        return label in self._labels(apoc_id)

    def _primary_label(self, apoc_id: str) -> str:
        labels = self._labels(apoc_id)
        return labels[0] if labels else "Node"

    def _prop(self, apoc_id: str, key: str):
        node = self._nodes.get(apoc_id)
        if not node:
            return None
        return _scalar((node["properties"] or {}).get(key))

    def _stable_id(self, apoc_id: str) -> str:
        pid = self._prop(apoc_id, "id")
        return str(pid) if pid else f"#{apoc_id}"

    def _app_id_of_pi(self, apoc_id: str) -> "str | None":
        for key in _APP_ID_KEYS:
            value = self._prop(apoc_id, key)
            if value not in (None, ""):
                return str(value)
        return None

    # ------------------------------------------------------------------ #
    #  ProcessInstance index + hops-bounded neighborhood                   #
    # ------------------------------------------------------------------ #
    def _build_pi_index(self) -> None:
        """Index ProcessInstance nodes by every id-form they carry.

        Applications are located and scoped purely by a hops-bounded walk from
        their ProcessInstance (:meth:`_neighborhood`) - there is deliberately NO
        connected-component tagging, because Actors are merged globally across
        applications and would glue unrelated ones together.
        """
        for apoc_id, node in self._nodes.items():
            if _PROCESS not in (node["labels"] or []):
                continue
            app_id = self._app_id_of_pi(apoc_id)
            if app_id is None:
                continue
            self._app_of_pi[apoc_id] = app_id
            bucket = self._pis_by_app.setdefault(app_id, [])
            if apoc_id not in bucket:
                bucket.append(apoc_id)
            for key in _APP_ID_KEYS:
                value = self._prop(apoc_id, key)
                if value not in (None, ""):
                    self._pi_by_key.setdefault(str(value), apoc_id)

    def _resolve_pis(self, application_id: str) -> list[str]:
        """ProcessInstance apoc ids for an application id (any id-form)."""
        application_id = str(application_id)
        pis = self._pis_by_app.get(application_id)
        if pis:
            return pis
        apoc_id = self._pi_by_key.get(application_id)
        return [apoc_id] if apoc_id else []

    def _seeds_for(self, pis: "list[str]") -> list[str]:
        """A case's ProcessInstance(s) PLUS the ActivityCluster(s) part_of them.

        The cluster sits between the ProcessInstance and all of the case's work,
        so seeding from the PI alone would push evidence and decisions out by one
        hop and rationales out of range entirely. Seeding both puts every node
        at the same distance it had when the work hung off the PI directly, so
        ``hops`` keeps its meaning.
        """
        seeds = list(pis)
        for pi in pis:
            for neighbour_id, _rel, _dir in self._adj.get(pi, []):
                if (
                    self._has_label(neighbour_id, _CLUSTER)
                    and neighbour_id not in seeds
                ):
                    seeds.append(neighbour_id)
        return seeds

    @property
    def exclude_same_case(self) -> bool:
        """Whether sibling occurrences of a case are barred from retrieval.

        Public so the retrieval tools can enforce the SAME policy on ids the
        agent supplies directly. Filtering the similarity ranking only hides
        sibling occurrences; it cannot stop a lookup by an id the agent picked
        up elsewhere, such as a ProcessInstance line in a rendered
        neighbourhood.
        """
        return self._exclude_same_case

    def is_same_base_case(self, a: str, b: str) -> bool:
        """True when two application ids are occurrences of one original case."""
        return self._base_case(a) == self._base_case(b)

    @staticmethod
    def _base_case(application_id: str) -> str:
        """Base (original) case id shared by every occurrence of a case.

        Occurrence ids are ``<base case>__<opaque token>``, so two occurrences
        of one case (``application_347__3f9c2d18b04a`` and
        ``application_347__b71e0a4c9d52``) reduce to ``application_347`` while
        the token itself says nothing about which run each one hides.

        This is why the occurrence token is a suffix rather than the whole id:
        the sibling guard in ``build_tools`` has to be able to recognise another
        sampling of the case under reconstruction, and this is how it does it.
        """
        return str(application_id).split("__", 1)[0]

    def _pis_reachable_from(self, apoc_id: str, hops: int = 4) -> set[str]:
        """ProcessInstances owning *apoc_id*, crossing the ActivityCluster.

        Deliberately NOT :meth:`_neighborhood`. That walk treats every boundary
        label as terminal, and the ActivityCluster is one -- so once the case's
        work moved onto the cluster, the only route a Decision/Rationale has to
        its ProcessInstance (``-> ActivityCluster -part_of-> ProcessInstance``)
        became untraversable and this map silently emptied. An empty map is not
        a degraded lookup, it disables BOTH similarity guards: the query's own
        application stops being excluded and ``exclude_same_case`` becomes a
        no-op, so a case can match another occurrence of itself.

        Crossing a cluster cannot leak into another application the way the
        render walk can, because a cluster is ``part_of`` exactly one
        ProcessInstance. The genuinely shared node type (Actor) stays terminal.
        """
        blocked = {_ACTOR}
        found: set[str] = set()
        seen = {apoc_id}
        frontier = {apoc_id}
        for _ in range(max(0, int(hops))):
            nxt: set[str] = set()
            for nid in frontier:
                for neighbour_id, _rel, _dir in self._adj.get(nid, []):
                    if neighbour_id in seen or neighbour_id not in self._nodes:
                        continue
                    seen.add(neighbour_id)
                    if neighbour_id in self._app_of_pi:
                        found.add(neighbour_id)
                        continue  # a ProcessInstance is where the walk ends
                    if any(self._has_label(neighbour_id, lab) for lab in blocked):
                        continue
                    nxt.add(neighbour_id)
            frontier = nxt
            if not frontier:
                break
        return found

    def _build_node_apps(self) -> None:
        """Tag each Decision/Rationale with its owning application + base case."""
        for apoc_id, node in self._nodes.items():
            labels = node["labels"] or []
            if _DECISION not in labels and _RATIONALE not in labels:
                continue
            apps = sorted({
                self._app_of_pi[n] for n in self._pis_reachable_from(apoc_id)
            })
            if apps:
                self._app_of_node[apoc_id] = apps[0]
                self._base_of_node[apoc_id] = self._base_case(apps[0])

    def _is_boundary(self, apoc_id: str) -> bool:
        return any(self._has_label(apoc_id, lab) for lab in _BOUNDARY_LABELS)

    def _neighborhood(self, seeds: "list[str] | set[str]", hops: int) -> set[str]:
        """Hops-bounded BFS from *seeds*.

        Boundary nodes (a different ProcessInstance, its ActivityCluster, or an
        Actor) are INCLUDED when reached but never expanded THROUGH, so the walk
        stays within one application instead of leaking into others via globally
        shared nodes. Event/Activity nodes are dropped unless ``include_event``
        is set.
        """
        keep_event = self._include_event
        # An Actor is never a seed: as a boundary node a seed still expands, and
        # a globally-merged Actor would pull in every case it ever touched.
        seed_set = {
            s for s in seeds
            if s in self._nodes and not self._has_label(s, _ACTOR)
        }
        seen: set[str] = set(seed_set)
        frontier: set[str] = set(seed_set)
        for _ in range(max(0, int(hops))):
            nxt: set[str] = set()
            for nid in frontier:
                # Seeds always expand; boundary nodes reached later are terminal.
                if nid not in seed_set and self._is_boundary(nid):
                    continue
                for neighbour_id, _rel, _dir in self._adj.get(nid, []):
                    if neighbour_id in seen or neighbour_id not in self._nodes:
                        continue
                    if not keep_event and any(
                        self._has_label(neighbour_id, lab) for lab in _LEAKY_LABELS
                    ):
                        continue
                    nxt.add(neighbour_id)
            seen |= nxt
            frontier = nxt
            if not frontier:
                break
        if not keep_event:
            seen = {
                n for n in seen
                if not any(self._has_label(n, lab) for lab in _LEAKY_LABELS)
            }
        return seen

    # ------------------------------------------------------------------ #
    #  Similarity matrices (fail-hard on missing embeddings)              #
    # ------------------------------------------------------------------ #
    def _build_label_matrices(self) -> None:
        missing: list[tuple[str, list[str]]] = []
        pending: dict[str, dict] = {}
        for label in (_DECISION, _RATIONALE):
            ids: list[str] = []
            apocs: list[str] = []
            text_vecs: list[list[float]] = []
            fastrp_vecs: list[list[float]] = []
            for apoc_id, node in self._nodes.items():
                if label not in (node["labels"] or []):
                    continue
                props = node["properties"] or {}
                t_vec = _as_vector(props.get(SEMANTIC_EMBED_PROPERTY))
                f_vec = _as_vector(props.get(FASTRP_EMBED_PROPERTY))
                lacking = []
                if t_vec is None:
                    lacking.append(SEMANTIC_EMBED_PROPERTY)
                if f_vec is None:
                    lacking.append(FASTRP_EMBED_PROPERTY)
                if lacking:
                    missing.append((self._stable_id(apoc_id), lacking))
                    continue
                ids.append(self._stable_id(apoc_id))
                apocs.append(apoc_id)
                text_vecs.append(t_vec)
                fastrp_vecs.append(f_vec)
            pending[label] = {
                "ids": ids, "apoc": apocs,
                "text": text_vecs, "fastrp": fastrp_vecs,
            }

        if missing:
            head = "; ".join(
                f"{nid} missing {'+'.join(props)}" for nid, props in missing[:20]
            )
            more = "" if len(missing) <= 20 else f" (+{len(missing) - 20} more)"
            raise ValueError(
                "graph.json has Decision/Rationale nodes without embeddings "
                f"({len(missing)} node(s)): {head}{more}. Re-run the "
                "embed_graph pass so every node carries both "
                f"{SEMANTIC_EMBED_PROPERTY} and {FASTRP_EMBED_PROPERTY}."
            )

        for label, data in pending.items():
            self._label_data[label] = self._finalize_label(label, data)

    def _finalize_label(self, label: str, data: dict) -> dict:
        ids = data["ids"]
        row_of = {nid: i for i, nid in enumerate(ids)}
        text = self._normalized_matrix(data["text"], label, SEMANTIC_EMBED_PROPERTY)
        fastrp = self._normalized_matrix(data["fastrp"], label, FASTRP_EMBED_PROPERTY)
        return {
            "ids": ids,
            "apoc": data["apoc"],
            "row_of": row_of,
            "text": text,
            "fastrp": fastrp,
            "app": np.array(
                [self._app_of_node.get(a, "") for a in data["apoc"]], dtype=object
            ),
            "base": np.array(
                [self._base_of_node.get(a, "") for a in data["apoc"]], dtype=object
            ),
        }

    @staticmethod
    def _normalized_matrix(vectors: list[list[float]], label: str, prop: str) -> np.ndarray:
        if not vectors:
            return np.zeros((0, 0), dtype=np.float64)
        matrix = np.asarray(vectors, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError(
                f"{label}.{prop} vectors have inconsistent dimensions; "
                "cannot build a similarity matrix."
            )
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / (norms + 1e-12)

    # ------------------------------------------------------------------ #
    #  Similarity queries                                                 #
    # ------------------------------------------------------------------ #
    def _similar(
        self,
        label: str,
        query_id: str,
        k: int,
        hops: int,
        *,
        current_app: "str | None" = None,
    ) -> "list[dict] | None":
        data = self._label_data.get(label)
        if data is None:
            return None
        row = data["row_of"].get(str(query_id))
        if row is None:
            return None  # unknown / wrong-label id -> tool reports "not found"
        text = data["text"]
        fastrp = data["fastrp"]
        n = len(data["ids"])
        if n <= 1:
            return []
        sem = text @ text[row]
        struct = fastrp @ fastrp[row]
        score = self._alpha * sem + (1.0 - self._alpha) * struct
        score[row] = -np.inf

        # Two DIFFERENT applications must be excluded, and conflating them is a
        # leak. The QUERY's application is excluded because its nodes are
        # redundant -- the agent already holds them. *current_app*, the case
        # actually being reconstructed, is excluded because its nodes and those
        # of its sibling occurrences are the ANSWER.
        #
        # They coincide only on a first-hop call seeded from the task. The agent
        # also chains: it takes a hit from another case and queries THAT node.
        # Guarding on the query alone then excludes the wrong case and hands the
        # reconstruction target straight back -- every sibling that survived the
        # same-case filter in practice arrived this way, none through a seed.
        for app in (self._app_of_node.get(data["apoc"][row]), current_app):
            if not app:
                continue
            score[data["app"] == app] = -np.inf
            if self._exclude_same_case:
                score[data["base"] == self._base_case(app)] = -np.inf

        k = max(1, int(k))
        order = np.argsort(-score)
        hits: list[dict] = []
        for j in order:
            if len(hits) >= k:
                break
            if not np.isfinite(score[j]):
                continue
            apoc_id = data["apoc"][j]
            hits.append({
                "id": data["ids"][j],
                "apoc": apoc_id,
                "score": float(score[j]),
                "semantic": float(sem[j]),
                "structural": float(struct[j]),
                "application_id": self._app_of_node.get(apoc_id, "unknown"),
                "text": _clip(self._prop(apoc_id, "text") or ""),
            })
        return hits

    def similar_decisions_text(
        self, decision_node_id: str, k: int, hops: int,
        *, current_app: "str | None" = None,
    ) -> str:
        """*current_app* is the case being reconstructed; its own occurrences are
        never returned, however far the agent has chained from its seeds."""
        return self._render_similar(
            _DECISION, "decision", decision_node_id, k, hops, current_app=current_app,
        )

    def similar_rationales_text(
        self, rationale_id: str, k: int, hops: int,
        *, current_app: "str | None" = None,
    ) -> str:
        """See :meth:`similar_decisions_text` for *current_app*."""
        return self._render_similar(
            _RATIONALE, "rationale", rationale_id, k, hops, current_app=current_app,
        )

    def _render_similar(
        self, label: str, noun: str, query_id: str, k: int, hops: int,
        *, current_app: "str | None" = None,
    ) -> str:
        hits = self._similar(label, query_id, k, hops, current_app=current_app)
        if hits is None:
            known = "a DecisionNode id (dec_...)" if label == _DECISION \
                else "a RationaleNode id (rat_...)"
            return (
                f"No {noun} with id {query_id!r} is in the graph. "
                f"Pass {known} taken from the current application's subgraph."
            )
        if not hits:
            return f"No other {noun}s are available to compare against."
        lines = [
            f"Top {len(hits)} {noun}s similar to {query_id} "
            f"(hybrid score = {self._alpha:.2f}*semantic + "
            f"{1 - self._alpha:.2f}*structural):",
            "",
        ]
        for rank, hit in enumerate(hits, start=1):
            lines.append(
                f"{rank}. [{hit['id']}] score={hit['score']:.3f} "
                f"(semantic={hit['semantic']:.3f}, structural={hit['structural']:.3f}) "
                f"application={hit['application_id']}"
            )
            if hit["text"]:
                lines.append(f'   text: "{hit["text"]}"')
            neighborhood = self._neighborhood([hit["apoc"]], hops)
            rendered = self._render_subgraph(neighborhood, focus={hit["apoc"]})
            if rendered:
                lines.append(f"   neighborhood (<= {hops} hop(s)):")
                lines.extend("   " + ln for ln in rendered.splitlines())
            lines.append("")
        return "\n".join(lines).rstrip()

    # ------------------------------------------------------------------ #
    #  Application subgraph / seeds                                        #
    # ------------------------------------------------------------------ #
    def has_application(self, application_id: str) -> bool:
        return bool(self._resolve_pis(application_id))

    def application_subgraph_text(
        self,
        application_id: str,
        hops: int,
        this_email: bool = False,
    ) -> str:
        pis = self._resolve_pis(application_id)
        if not pis:
            return (
                f"No application with id {application_id!r} is in the graph "
                f"(no matching {_PROCESS})."
            )
        neighborhood = self._neighborhood(self._seeds_for(pis), hops)
        rendered = self._render_subgraph(
            neighborhood, focus=set(pis), this_email=this_email
        )
        header = f"Context subgraph for application {application_id} (<= {hops} hop(s)):"
        return f"{header}\n{rendered}" if rendered else header

    def application_seed_ids(self, application_id: str, hops: int = 2) -> dict:
        """Decision / rationale / evidence ids within *hops* of the app's PI.

        Used to seed the agent's opening context with valid ids to expand. Scoped
        by the same hops-bounded neighborhood as the subgraph, so it stays within
        one application (no connected-component bleed across shared value nodes).
        """
        out: dict[str, list[str]] = {"decisions": [], "rationales": [], "evidence": []}
        pis = self._resolve_pis(application_id)
        if not pis:
            return out
        for apoc_id in self._neighborhood(self._seeds_for(pis), hops):
            label = self._primary_label(apoc_id)
            if label == _DECISION:
                out["decisions"].append(self._stable_id(apoc_id))
            elif label == _RATIONALE:
                out["rationales"].append(self._stable_id(apoc_id))
            elif label == _EVIDENCE:
                out["evidence"].append(self._stable_id(apoc_id))
        for key in out:
            out[key] = sorted(out[key])
        return out

    # ------------------------------------------------------------------ #
    #  Rendering                                                          #
    # ------------------------------------------------------------------ #
    def _span_index(self, apoc_id: str) -> int:
        """1-based position of a body-span EvidenceNode in the email (0 if none).

        Body-span EvidenceNode ids end in ``__s<index>`` (reading order), so the
        span index is the order in which the sender narrates the steps - the only
        ordering signal the reconstruction should follow.
        """
        match = _SPAN_INDEX_RE.search(self._stable_id(apoc_id))
        return int(match.group(1)) if match else 0

    def _node_details(self, apoc_id: str) -> str:
        props = self._nodes[apoc_id]["properties"] or {}
        parts: list[str] = []
        if self._has_label(apoc_id, _EVIDENCE):
            span = self._span_index(apoc_id)
            if span:
                parts.append(f"span={span}")
        for key in _PREF_ORDER:
            if key in _SKIP_KEYS:
                continue
            value = _scalar(props.get(key))
            if value in (None, ""):
                continue
            text = _clip(str(value))
            if key == "text":
                parts.append(f'"{text}"')
            elif key == "name":
                parts.append(text)
            else:
                parts.append(f"{key}={text}")
        return " | ".join(parts) if parts else "no details"

    def _drop_bare_process_instances(self, apoc_ids: set[str]) -> set[str]:
        """Drop ProcessInstance nodes with no directly-connected Evidence/Decision/
        Rationale in *apoc_ids* (general rule: don't surface a bare application).
        """
        ids = set(apoc_ids)
        kept: set[str] = set()
        for nid in ids:
            if self._has_label(nid, _PROCESS):
                linked = any(
                    nb in ids
                    and any(self._has_label(nb, lab) for lab in _PI_LINK_LABELS)
                    for nb, _rel, _dir in self._adj.get(nid, [])
                )
                if not linked:
                    continue
            kept.add(nid)
        return kept

    def _render_subgraph(
        self,
        apoc_ids: set[str],
        focus: "set[str] | None" = None,
        this_email: bool = False,
    ) -> str:
        """Render via the shared island renderer, so every prompt shows the same
        ``Node:`` / ``Relationships:`` shape as the non-agentic arm.

        ``this_email`` marks the rendered EvidenceNodes as belonging to the email
        under reconstruction; another case's spans render as ``other``.
        """
        if not apoc_ids:
            return ""
        apoc_ids = self._drop_bare_process_instances(apoc_ids)
        if not apoc_ids:
            return ""
        anchors = (
            {n for n in apoc_ids if self._has_label(n, _EVIDENCE)}
            if this_email else frozenset()
        )
        return render_island(
            apoc_ids,
            primary_label=self._primary_label,
            props_of=lambda nid: self._nodes[nid]["properties"],
            span_index_of=self._span_index,
            out_edges_of=lambda nid: [
                (dst, rel)
                for dst, rel, direction in self._adj.get(nid, [])
                if direction == "out"
            ],
            anchors=anchors,
            mask_execution_verbs=self._mask_execution_verbs,
        )
