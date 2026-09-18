#!/usr/bin/env python3
"""CLI entry point: reconstruct hidden emails (Pass B - predict).

It reads the generator's ``generated.jsonl`` from ``--emails`` and predicts, for
each hidden email, the ORDERED activity sequence it reports, writing
``predicted.jsonl`` and one ``<graph_key>/llm_prompts.md`` per email to ``--out``. Each hidden email is reconstructed
INDEPENDENTLY from the OFFLINE observed timeline (past + future, co-dropped
emails hidden). The prediction length is NOT capped - the model is never told
how many activities are hidden, so we can measure over-/under-prediction.

Knobs:
  --use-email          : show the email's subject/body/sender/objects.
  --use-ocel           : append the offline observed case timeline (past +
                         future, hidden email marked THIS).
  --use-graph          : append the case subgraph around the email's
                         EvidenceNode (from Prepare -> Ingest).
  --use-agentic-tools  : (graph arm only) let the agent call the retrieval tools
                         (default). Set false to answer single-shot from the
                         cold-start context (subgraph + timeline) with no tools.
  --filter-event-node  : drop the activity-leaking ``Event`` / ``Activity``
                         nodes from that subgraph (the "clean" variant).
  --mask-execution-verbs : (graph arm only, default true) collapse the six Actor
                         execution verbs pointing at the case's
                         ``ActivityCluster`` to a neutral ``linked_to`` edge.
  --reasoning-effort   : gpt-5.x reasoning budget (default ``high``) applied to
                         both the single-shot predictor and the agent.

The activity catalog is read from the ``--process-workdir`` folder (the same
``schema_<name>_only.json`` the email generator uses), selected by
``--schema-name``. The Azure OpenAI API key is read from the
``AZURE_OPENAI_API_KEY`` environment variable (never a command-line argument);
absent that, auth falls back to a ``DefaultAzureCredential`` bearer token.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make the sibling ``src/`` importable regardless of the launch cwd.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


_TRUE = ("1", "true", "yes", "on", "y", "t")
_FALSE = ("0", "false", "no", "off", "n", "f")


def _str2bool(value: str, *, flag: str) -> bool:
    """Parse a component boolean, raising on anything unrecognised.

    These flags are typed by hand on the command line, and every one of
    them changes what the experiment measures. Treating an unrecognised string
    as False -- which is what ``value in _TRUE`` did -- means a typo silently
    inverts a run's configuration and the only trace is the config line in a
    log nobody reads. That is exactly how ``exclude_same_case`` was submitted as
    ``"ture"`` and ran with the same-case guard off.
    """
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ValueError(
        f"{flag}: {value!r} is not a boolean. Use one of "
        f"{', '.join(_TRUE)} / {', '.join(_FALSE)}."
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reconstruct hidden emails (size/count experiment)."
    )
    p.add_argument("--emails", required=True,
                   help="Folder holding the generator's generated.jsonl.")
    p.add_argument("--graphs", default="",
                   help="Folder holding the whole ingested graph.json. Used only "
                        "when --use-graph is true.")
    p.add_argument("--out", required=True,
                   help="Output folder; predicted.jsonl is written here.")
    p.add_argument("--process-workdir", required=True,
                   help="Folder holding schema_<name>_only.json (the activity catalog), "
                        "same as the OCEL Email Generator's process_workdir.")
    p.add_argument("--schema-name", default="bank_loan",
                   help="Selects schema_<schema-name>_only.json inside process_workdir.")
    p.add_argument("--schema", default="",
                   help="Explicit schema json path (overrides --process-workdir/--schema-name).")
    p.add_argument("--use-graph", default="true",
                   help="Append the EvidenceNode island to the prompt.")
    p.add_argument("--use-agentic-tools", default="true",
                   help="Graph arm only: if true (default) the agent calls the "
                        "retrieval tools; if false it answers single-shot from the "
                        "cold-start context (subgraph + timeline), no tools.")
    p.add_argument("--filter-event-node", default="true",
                   help="Drop the activity-leaking Event neighbours from the graph context.")
    p.add_argument("--mask-execution-verbs", default="true",
                   help="Graph arm only: collapse the six Actor execution verbs "
                        "pointing at the case's ActivityCluster to a neutral "
                        "linked_to edge (default). Set false to render the real "
                        "verbs, which hint at the kind of activity behind them.")
    p.add_argument("--use-email", default="true",
                   help="Show the email content to the model.")
    p.add_argument("--use-ocel", default="true",
                   help="Append the offline observed case timeline to the prompt.")
    p.add_argument("--sim-alpha", type=float, default=0.5,
                   help="Hybrid similarity weight: alpha*semantic + (1-alpha)*structural.")
    p.add_argument("--default-k", type=int, default=5,
                   help="Default number of similar decisions/rationales a tool returns.")
    p.add_argument("--default-hops", type=int, default=1,
                   help="Default neighborhood radius (hops) returned by the retrieval tools.")
    p.add_argument("--max-tool-iterations", type=int, default=6,
                   help="Max sequential tool-call iterations the agent may perform per email.")
    p.add_argument("--agent-model", default="gpt-5.4",
                   help="Model name reported to AutoGen (the Azure deployment that serves it "
                        "is set via --azure-deployment).")
    p.add_argument("--reasoning-effort", default="high",
                   help="gpt-5.x reasoning budget (minimal|low|medium|high) used by "
                        "both the single-shot predictor and the agent. Empty leaves "
                        "the model default.")
    p.add_argument("--email-activity", default="internal_email_sent",
                   help="OCEL event type of the placeholder email event.")
    p.add_argument("--exclude-same-case", default="true",
                   help="If true, find_similar_* assigns 0 (excludes) any decision/"
                        "rationale from another occurrence of the SAME original case. "
                        "On by default: a sibling occurrence hides a different email, "
                        "so its observed timeline shows this case's answer.")
    p.add_argument("--workers", type=int, default=8,
                   help="Parallel prediction requests (1 = sequential).")
    # Azure OpenAI wiring (endpoint/deployment/api-version -> env for the client).
    p.add_argument("--azure-endpoint", default="")
    p.add_argument("--azure-deployment", default="")
    p.add_argument("--azure-api-version", default="")
    return p.parse_args()


def _export_azure_env(args: argparse.Namespace) -> None:
    """Push the Azure OpenAI wiring into the env the client reads."""
    if args.azure_endpoint:
        os.environ["AZURE_OPENAI_ENDPOINT"] = args.azure_endpoint
    if args.azure_deployment:
        os.environ["AZURE_OPENAI_DEPLOYMENT"] = args.azure_deployment
    if args.azure_api_version:
        os.environ["AZURE_OPENAI_API_VERSION"] = args.azure_api_version


def _resolve_schema(workdir: Path, schema_name: str) -> Path:
    """Locate the activity catalog inside process_workdir (mirrors the generator).

    Prefers ``schema_<schema_name>_only.json``; falls back to a name-matched or
    the sole ``schema_*_only.json`` / ``schema_*.json`` present.
    """
    preferred = workdir / f"schema_{schema_name}_only.json"
    if preferred.exists():
        return preferred
    for pattern in (f"schema_{schema_name}*.json", "schema_*_only.json", "schema_*.json"):
        matches = sorted(workdir.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"No schema found in {workdir} (expected schema_{schema_name}_only.json)."
    )


def main() -> None:
    args = _parse_args()
    _export_azure_env(args)
    use_graph = _str2bool(args.use_graph, flag="--use-graph")
    use_agentic_tools = _str2bool(args.use_agentic_tools, flag="--use-agentic-tools")
    filter_event_node = _str2bool(args.filter_event_node, flag="--filter-event-node")
    mask_execution_verbs = _str2bool(
        args.mask_execution_verbs, flag="--mask-execution-verbs",
    )
    use_email = _str2bool(args.use_email, flag="--use-email")
    use_ocel = _str2bool(args.use_ocel, flag="--use-ocel")
    exclude_same_case = _str2bool(args.exclude_same_case, flag="--exclude-same-case")

    emails_root = Path(args.emails)
    graphs_root = Path(args.graphs) if args.graphs else None
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    if not os.environ.get("AZURE_OPENAI_API_KEY"):
        print("WARNING: AZURE_OPENAI_API_KEY not set; falling back to managed-identity auth.",
              flush=True)

    schema_path = (
        Path(args.schema) if args.schema
        else _resolve_schema(Path(args.process_workdir), args.schema_name)
    )
    print(f"Using activity catalog: {schema_path}", flush=True)

    from src.experiment import run_predict

    inp = emails_root / "generated.jsonl"
    if not inp.exists():
        matches = sorted(emails_root.glob("generated*.jsonl"))
        if not matches:
            raise SystemExit(
                f"No generated.jsonl (nor generated*.jsonl) found under {emails_root}."
            )
        inp = matches[0]
    n_scored, out_path = run_predict(
        inp_path=inp,
        out_path=out_root / "predicted.jsonl",
        schema_path=schema_path,
        workers=args.workers,
        use_graph=use_graph,
        graphs_root=(graphs_root if use_graph else None),
        filter_event_node=filter_event_node,
        mask_execution_verbs=mask_execution_verbs,
        use_email=use_email,
        use_ocel=use_ocel,
        sim_alpha=args.sim_alpha,
        default_k=args.default_k,
        default_hops=args.default_hops,
        max_tool_iterations=args.max_tool_iterations,
        agent_model=args.agent_model,
        email_activity=args.email_activity,
        exclude_same_case=exclude_same_case,
        reasoning_effort=args.reasoning_effort.strip(),
        use_agentic_tools=use_agentic_tools,
    )
    print(f"\nDone: {n_scored} email(s) predicted -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
