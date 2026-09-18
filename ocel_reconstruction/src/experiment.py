"""Experiment reconstruction (Pass B - predict), embedded here.

Reads the JSONL a Pass A generator wrote (``generated.jsonl``) and reconstructs
each hidden email INDEPENDENTLY. Three knobs select the prompt: ``use_email``
(the email payload), ``use_ocel`` (the offline observed timeline, past + future,
co-dropped emails hidden), and ``use_graph`` (the case subgraph around the
email's EvidenceNode). The prediction length is NOT capped: the model is
never told how many activities are hidden, so we can observe whether it
over-/under-predicts (hallucinates). ``filter_event_node`` drops the
activity-leaking Event/Activity nodes (no leakage).

``use_graph=true`` always runs the agentic path; the non-graph path is the
single-shot ``SequencePredictor`` on the email and/or timeline alone.

Every input record is echoed to the output JSONL with two added fields, ``pred``
(the predicted ordered activity list) and ``confidence``; the evaluation
component then scores ``true_members`` vs ``pred``. Each hidden email also gets
its own ``<graph_key>/llm_prompts.md`` (system/user/assistant exchange), matching
the email generator's per-occurrence prompt dump.

The prediction runs against the FULL activity catalog (never the email-capable
restriction), because a dropped unit may be an activity that is never itself the
subject of an email - restricting the catalog would make it unreconstructable.
"""
from __future__ import annotations

import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.azure_client import FatalAzureError
from src.sequence_predictor import SequencePredictor, build_allowed, build_catalog


def _objects_str(pipe_objects: str) -> str:
    return ", ".join(p.strip() for p in (pipe_objects or "").split("|") if p.strip())


def _md_fence(text: str, lang: str = "") -> str:
    """Fenced code block that stays valid if *text* already contains backticks."""
    fence = "```"
    while fence in text:
        fence += "`"
    return f"{fence}{lang}\n{text}\n{fence}"


def format_llm_prompts_markdown(
    llm_calls: list[dict],
    *,
    title: str = "LLM prompts",
) -> str:
    """Render captured LLM exchanges as a readable markdown document.

    Same shape as the email generator's ``llm_prompts.md``: one section per call
    with system / user / assistant fences.
    """
    lines = [f"# {title}", ""]
    if not llm_calls:
        lines.append("_No LLM calls were made for this run._")
        lines.append("")
        return "\n".join(lines)

    for i, call in enumerate(llm_calls, start=1):
        email_id = call.get("email_id") or "?"
        pass_name = call.get("pass") or "predict"
        status = "ok" if call.get("ok") else "failed"
        if call.get("cached"):
            status = "cache-hit"
        heading = f"## {i}. `{email_id}` — {pass_name}"
        heading += f" [{status}]"
        lines.append(heading)
        lines.append("")
        if call.get("error"):
            lines.append(f"**Error:** {call['error']}")
            lines.append("")
        if call.get("note"):
            lines.append(f"**Note:** {call['note']}")
            lines.append("")
        if call.get("max_completion_tokens") is not None:
            lines.append(f"**max_completion_tokens:** {call['max_completion_tokens']}")
            lines.append("")
        if call.get("reasoning_effort"):
            lines.append(f"**reasoning_effort:** {call['reasoning_effort']}")
            lines.append("")
        if call.get("system"):
            lines.append("### System")
            lines.append("")
            lines.append(_md_fence(str(call["system"])))
            lines.append("")
        if call.get("user"):
            lines.append("### User")
            lines.append("")
            lines.append(_md_fence(str(call["user"])))
            lines.append("")
        if call.get("assistant") is not None:
            lines.append("### Assistant")
            lines.append("")
            lines.append(_md_fence(str(call["assistant"])))
            lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


def _prompt_folder_name(record: dict, idx: int) -> str:
    """Filesystem-safe folder name for one email's prompt markdown."""
    raw = record.get("graph_key") or record.get("email_uid") or f"record_{idx:04d}"
    name = str(raw).strip().replace("/", "_").replace("\\", "_")
    return name or f"record_{idx:04d}"


def _strip_ocel_generator_header(timeline: str) -> str:
    """Drop the generator's 'use BOTH sides' header; keep THIS marker and events."""
    kept: list[str] = []
    skipping_blanks = True
    for line in timeline.splitlines():
        if "use BOTH sides" in line or line.startswith("OBSERVED CASE TIMELINE"):
            continue
        if skipping_blanks and not line.strip():
            continue
        skipping_blanks = False
        kept.append(line)
    return "\n".join(kept).strip()


def _ocel_text_for(record: dict, use_ocel: bool) -> str:
    """Observed (redacted) case timeline for a record, or "" when withheld.

    An empty string means the OCEL source block is omitted from the prompt.
    Used by both paths: the single-shot predictor and the agent's opening
    context.
    """
    if not use_ocel:
        return ""
    timeline = record.get("observed_timeline")
    if timeline is None:
        raise ValueError(
            f"record {record.get('email_uid') or record.get('graph_key')!r} "
            "is missing observed_timeline"
        )
    if str(timeline).strip():
        return _strip_ocel_generator_header(str(timeline))
    return ""


def run_predict(
    inp_path: str | Path,
    out_path: str | Path,
    schema_path: str | Path,
    workers: int = 8,
    use_graph: bool = False,
    graphs_root: "str | Path | None" = None,
    filter_event_node: bool = True,
    mask_execution_verbs: bool = True,
    use_email: bool = True,
    use_ocel: bool = True,
    sim_alpha: float = 0.5,
    default_k: int = 5,
    default_hops: int = 1,
    max_tool_iterations: int = 6,
    agent_model: str = "gpt-5.4",
    email_activity: str = "internal_email_sent",
    exclude_same_case: bool = False,
    reasoning_effort: str = "high",
    use_agentic_tools: bool = True,
) -> tuple[int, Path]:
    """Predict every hidden email in *inp_path*, writing *out_path* JSONL.

    Three independent knobs select what the model sees:
      * ``use_email`` — subject/body/sender/objects
      * ``use_ocel`` — the offline observed case timeline (past + future, with
        the hidden email marked THIS)
      * ``use_graph`` — the case subgraph around the email's EvidenceNode(s),
        from the whole ingested graph (``graphs_root/graph.json``);
        ``filter_event_node`` drops the activity-leaking Event/Activity nodes

    ``mask_execution_verbs`` (graph arm only, default True) collapses the six
    Actor execution verbs pointing at the case's ActivityCluster to a neutral
    ``linked_to``. Each verb is an LLM's one-word summary of what an actor did,
    so it hints at the kind of activity behind it; masking keeps the actor
    attached to the case without that hint.

    ``reasoning_effort`` ("minimal" | "low" | "medium" | "high", default "high")
    is the gpt-5.x reasoning budget used by BOTH paths — the single-shot
    predictor and the agentic one. An empty string leaves the model default.

    ``use_agentic_tools`` (default True) only applies to the graph arm
    (``use_graph=true``). When False the graph config still assembles the case
    subgraph + timeline into the prompt (the agent's "cold start" context) but
    binds NO retrieval tools and skips the similarity-lookup retry, so the model
    answers single-shot from what it already has. It has no effect when
    ``use_graph=false`` (that arm is already single-shot).

    Also writes one ``<graph_key>/llm_prompts.md`` per predicted email next to
    *out_path* (same layout as the email generator).

    Returns ``(n_scored, out_path)``.
    """
    records = [
        json.loads(line)
        for line in Path(inp_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"loaded {len(records)} record(s) from {inp_path}", flush=True)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    catalog = build_catalog(schema)
    allowed = build_allowed(schema)
    print(f"catalog: {len(schema)} activities from {Path(schema_path).name} "
          f"(full schema; restrict-catalog N/A in experiment mode)", flush=True)

    # Graph-enabled configs run the agentic (AutoGen) path with retrieval tools;
    # non-graph configs keep the single-shot SequencePredictor below unchanged.
    # A single knob, use_email, governs email visibility on both paths: the
    # non-graph path shows the email block in the prompt; the graph path shows
    # (or, when false, redacts) the email inside the get_ocel_of_application tool.
    if use_graph:
        return asyncio.run(_run_predict_agentic(
            records=records,
            emails_root=Path(inp_path).parent,
            out_path=out_path,
            catalog=catalog,
            allowed=allowed,
            graphs_root=graphs_root,
            filter_event_node=filter_event_node,
            mask_execution_verbs=mask_execution_verbs,
            use_ocel=use_ocel,
            use_email=use_email,
            sim_alpha=sim_alpha,
            default_k=default_k,
            default_hops=default_hops,
            max_tool_iterations=max_tool_iterations,
            agent_model=agent_model,
            email_activity=email_activity,
            exclude_same_case=exclude_same_case,
            reasoning_effort=reasoning_effort,
            use_agentic_tools=use_agentic_tools,
            workers=workers,
        ))

    # Reaching here means use_graph is false: no graph source, single-shot only.
    predictor = SequencePredictor(
        force_single_activity=False, reasoning_effort=reasoning_effort,
    )
    print(f"use_email={use_email} use_ocel={use_ocel} "
          f"reasoning_effort={reasoning_effort or '(model default)'}", flush=True)

    # With the email shown we need it present; without it, every record with an
    # observed timeline is predictable (timeline-only baseline).
    if use_email:
        predictable = [(i, r) for i, r in enumerate(records) if r.get("email")]
    else:
        predictable = [(i, r) for i, r in enumerate(records)]
    preds: dict[int, dict] = {}
    llm_calls_by_idx: dict[int, dict] = {}

    def _one(idx: int, r: dict) -> tuple[int, list[str], float, dict]:
        email = r.get("email") or {}
        show = use_email and bool(r.get("email"))
        activities, confidence, _reason, call = predictor.predict_safe(
            subject=email.get("subject", "") if show else "",
            body=email.get("body", "") if show else "",
            sender=email.get("sender", "") if show else "",
            objects=_objects_str(email.get("objects", "")) if show else "",
            catalog=catalog,
            allowed=allowed,
            ocel_text=_ocel_text_for(r, use_ocel),
            # No length hint: the model is never told how many activities are
            # hidden, so we can observe whether it over-/under-predicts.
            max_activities=0,
        )
        call = {
            **call,
            "email_id": r.get("graph_key") or r.get("email_uid") or f"record_{idx}",
        }
        return idx, activities, confidence, call

    n = len(predictable)
    workers = max(1, min(workers, n)) if n else 1
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_one, i, r): i for i, r in predictable}
            try:
                done = 0
                for fut in as_completed(futures):
                    idx, activities, confidence, call = fut.result()
                    preds[idx] = {"pred": activities, "confidence": confidence}
                    llm_calls_by_idx[idx] = call
                    done += 1
                    if done % 50 == 0:
                        print(f"  predicted {done}/{n}", flush=True)
            except FatalAzureError:
                for f in futures:
                    f.cancel()
                raise
    else:
        for done, (i, r) in enumerate(predictable, start=1):
            idx, activities, confidence, call = _one(i, r)
            preds[idx] = {"pred": activities, "confidence": confidence}
            llm_calls_by_idx[idx] = call
            if done % 50 == 0:
                print(f"  predicted {done}/{n}", flush=True)

    n_scored = 0
    n_md = 0
    out_dir = out_path.parent
    with out_path.open("w", encoding="utf-8") as fout:
        for i, r in enumerate(records):
            p = preds.get(i, {"pred": [], "confidence": 0.0})
            fout.write(json.dumps({**r, **p}, ensure_ascii=False) + "\n")
            if i in preds:
                n_scored += 1
            call = llm_calls_by_idx.get(i)
            if call is None:
                continue
            folder_name = _prompt_folder_name(r, i)
            folder = out_dir / folder_name
            folder.mkdir(parents=True, exist_ok=True)
            md = format_llm_prompts_markdown(
                [call],
                title=f"LLM prompts — {folder_name}",
            )
            (folder / "llm_prompts.md").write_text(md, encoding="utf-8")
            n_md += 1

    print(f"Pass B predict complete: {n_scored} email(s) predicted -> {out_path}",
          flush=True)
    print(f"LLM prompts written -> {out_dir}/<graph_key>/llm_prompts.md ({n_md} file(s))",
          flush=True)
    return n_scored, out_path


# --------------------------------------------------------------------------- #
#  Agentic path (use_graph=true): AutoGen agent + retrieval tools               #
# --------------------------------------------------------------------------- #
def format_agent_transcript_markdown(call: dict, *, title: str = "Agentic run") -> str:
    """Render one agent run (system, task, messages + tool calls, final answer)."""
    lines = [f"# {title}", ""]
    status = "ok" if call.get("ok") else "failed"
    lines.append(f"**status:** {status}")
    lines.append("")
    if call.get("error"):
        lines.append(f"**Error:** {call['error']}")
        lines.append("")
    if call.get("max_tool_iterations") is not None:
        lines.append(f"**max_tool_iterations:** {call['max_tool_iterations']}")
        lines.append("")
    if call.get("reasoning_effort"):
        lines.append(f"**reasoning_effort:** {call['reasoning_effort']}")
        lines.append("")
    if call.get("system"):
        lines += ["## System", "", _md_fence(str(call["system"])), ""]
    if call.get("user"):
        lines += ["## Task", "", _md_fence(str(call["user"])), ""]
    transcript = call.get("transcript") or []
    if transcript:
        lines += ["## Agent run", ""]
        for i, msg in enumerate(transcript, start=1):
            lines.append(f"### {i}. {msg.get('source', '?')} [{msg.get('type', '')}]")
            lines.append("")
            lines.append(_md_fence(str(msg.get("content", ""))))
            lines.append("")
    if call.get("assistant") is not None:
        lines += ["## Final answer", "", _md_fence(str(call["assistant"])), ""]
    return "\n".join(lines)


# Below this share of resolvable records the graph arm is not measuring the
# graph any more, so refuse to burn a full run producing a mislabelled baseline.
_MIN_GRAPH_COVERAGE = float(os.getenv("RECON_MIN_GRAPH_COVERAGE", "0.9"))


def _check_graph_coverage(graph_index, records: list[dict], graph_file: Path) -> None:
    """Fail unless the graph actually holds the applications we must reconstruct.

    A graph that resolves no ProcessInstance still produces a complete, green
    run: every prompt simply carries "no matching ProcessInstance" instead of the
    case's spans, and the agent answers from the timeline alone. That is a
    graph-less baseline wearing the graph arm's label, so stop instead.
    """
    keys = [
        str(r.get("graph_key") or r.get("case_id") or "")
        for r in records
    ]
    resolved = sum(1 for k in keys if k and graph_index.has_application(k))
    total = len(keys)
    coverage = resolved / total if total else 0.0
    print(f"graph coverage: {resolved}/{total} record(s) resolve to a "
          f"ProcessInstance ({coverage:.1%})", flush=True)
    if resolved == 0:
        missing = ", ".join(k for k in keys[:3] if k)
        raise ValueError(
            f"{graph_file} resolves NONE of the {total} record(s) to a "
            f"ProcessInstance (e.g. {missing}). The graph is empty, truncated, "
            "or built from a different dataset; every prompt would carry no "
            "case evidence at all. Re-run the context-graph ingest for this "
            "email set (a reused ingest step will keep serving the same bad "
            "output) and check that its graph.json is non-empty."
        )
    if coverage < _MIN_GRAPH_COVERAGE:
        raise ValueError(
            f"{graph_file} resolves only {resolved}/{total} record(s) "
            f"({coverage:.1%}), below the required {_MIN_GRAPH_COVERAGE:.1%}. "
            "The graph does not cover this email set; re-run the ingest, or "
            "lower RECON_MIN_GRAPH_COVERAGE if a partial graph is intended."
        )


async def _run_predict_agentic(
    *,
    records: list[dict],
    emails_root: Path,
    out_path: Path,
    catalog: str,
    allowed: dict,
    graphs_root: "str | Path | None",
    filter_event_node: bool,
    mask_execution_verbs: bool,
    use_ocel: bool,
    use_email: bool,
    sim_alpha: float,
    default_k: int,
    default_hops: int,
    max_tool_iterations: int,
    agent_model: str,
    email_activity: str,
    exclude_same_case: bool,
    reasoning_effort: str,
    use_agentic_tools: bool = True,
    workers: int,
) -> tuple[int, Path]:
    """Reconstruct every hidden email with the AutoGen agent.

    With *use_agentic_tools* the agent calls the retrieval tools; without it the
    agent answers single-shot from its cold-start context (subgraph + timeline).
    """
    from src.agent_predictor import AgenticPredictor
    from src.graph_index import GraphIndex
    from src.ocel_lookup import OcelStore

    if not graphs_root:
        raise ValueError(
            "use_graph=true requires the graphs input (graph.json) for the "
            "agentic reconstruction path."
        )
    graphs_dir = Path(graphs_root)
    graph_file = graphs_dir / "graph.json"
    if not graph_file.exists():
        matches = sorted(graphs_dir.glob("*.json"))
        if not matches:
            raise FileNotFoundError(
                f"No graph.json (nor *.json) found under {graphs_dir}."
            )
        graph_file = matches[0]

    print(f"Building GraphIndex from {graph_file} "
          f"(alpha={sim_alpha}, filter_event_node={filter_event_node}, "
          f"exclude_same_case={exclude_same_case}, "
          f"mask_execution_verbs={mask_execution_verbs})...", flush=True)
    graph_index = GraphIndex(
        graph_file, alpha=sim_alpha, filter_event_node=filter_event_node,
        exclude_same_case=exclude_same_case,
        mask_execution_verbs=mask_execution_verbs,
    )
    _check_graph_coverage(graph_index, records, graph_file)
    ocel_store = OcelStore(
        emails_root, records, use_email=use_email, email_activity=email_activity,
    )
    predictor = AgenticPredictor(
        graph_index, ocel_store,
        model=agent_model,
        default_k=default_k,
        default_hops=default_hops,
        max_tool_iterations=max_tool_iterations,
        use_tools=use_agentic_tools,
        reasoning_effort=reasoning_effort,
        mask_execution_verbs=mask_execution_verbs,
    )
    mode = "agentic (tools)" if use_agentic_tools else "graph single-shot (no tools)"
    print(f"graph reconstruction [{mode}]: model={agent_model} use_ocel={use_ocel} "
          f"use_email={use_email} use_agentic_tools={use_agentic_tools} "
          f"mask_execution_verbs={mask_execution_verbs} "
          f"k={default_k} hops={default_hops} "
          f"max_tool_iterations={max_tool_iterations if use_agentic_tools else 'n/a'} "
          f"reasoning_effort={reasoning_effort or '(model default)'} "
          f"workers={workers}", flush=True)

    predictable = list(enumerate(records))
    n = len(predictable)
    preds: dict[int, dict] = {}
    calls: dict[int, dict] = {}
    sem = asyncio.Semaphore(max(1, workers))

    async def _one(idx: int, r: dict) -> tuple[int, list[str], float, dict]:
        async with sem:
            ocel_text = _ocel_text_for(r, use_ocel)
            activities, confidence, _reason, call = await predictor.predict(
                r, catalog, allowed, ocel_text=ocel_text,
            )
            call = {
                **call,
                "email_id": r.get("graph_key") or r.get("email_uid") or f"record_{idx}",
            }
            return idx, activities, confidence, call

    try:
        tasks = [asyncio.create_task(_one(i, r)) for i, r in predictable]
        done = 0
        for fut in asyncio.as_completed(tasks):
            idx, activities, confidence, call = await fut
            preds[idx] = {"pred": activities, "confidence": confidence}
            calls[idx] = call
            done += 1
            if done % 20 == 0:
                print(f"  reconstructed {done}/{n}", flush=True)
    finally:
        await predictor.close()

    n_scored = 0
    n_md = 0
    out_dir = out_path.parent
    with out_path.open("w", encoding="utf-8") as fout:
        for i, r in enumerate(records):
            p = preds.get(i, {"pred": [], "confidence": 0.0})
            fout.write(json.dumps({**r, **p}, ensure_ascii=False) + "\n")
            if i in preds:
                n_scored += 1
            call = calls.get(i)
            if call is None:
                continue
            folder_name = _prompt_folder_name(r, i)
            folder = out_dir / folder_name
            folder.mkdir(parents=True, exist_ok=True)
            md = format_agent_transcript_markdown(
                call, title=f"Agentic reconstruction — {folder_name}",
            )
            (folder / "llm_prompts.md").write_text(md, encoding="utf-8")
            n_md += 1

    print(f"Pass B predict (agentic) complete: {n_scored} email(s) predicted -> "
          f"{out_path}", flush=True)
    print(f"Agent transcripts written -> {out_dir}/<graph_key>/llm_prompts.md "
          f"({n_md} file(s))", flush=True)
    return n_scored, out_path
