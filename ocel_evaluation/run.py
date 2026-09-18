#!/usr/bin/env python3
"""CLI entry point: score reconstructed hidden emails (Pass B - score).

It reads the reconstruction's ``predicted.jsonl`` from ``--reconstructed`` and,
for every hidden email, scores ``true_members`` vs ``pred`` (Levenshtein,
multiset PRF, sequence PRF via LCS/ROUGE-L, Jaccard, alignment-based confusion
matrix, ...). Results are written to
``summary.{json,md}`` (overall plus breakdowns by size n and dropped unit, plus
a true×pred confusion matrix with insert/delete) plus ``scored_rows.jsonl``.

No LLM / Azure / network: pure stdlib.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the sibling ``src/`` importable regardless of the launch cwd.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Score reconstructed hidden emails."
    )
    p.add_argument("--reconstructed", required=True,
                   help="Folder holding the reconstruction's predicted.jsonl.")
    p.add_argument("--out", required=True,
                   help="Output folder; summary.{json,md} + scored_rows.jsonl "
                        "are written here.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    recon_root = Path(args.reconstructed)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    from src.experiment import run_aggregate

    inp = recon_root / "predicted.jsonl"
    if not inp.exists():
        matches = sorted(recon_root.glob("predicted*.jsonl"))
        if not matches:
            raise SystemExit(
                f"No predicted.jsonl (nor predicted*.jsonl) found under {recon_root}."
            )
        inp = matches[0]
    n_scored, out_json = run_aggregate(inp, out_root)
    print(f"\nDone: scored {n_scored} email(s) -> {out_json}", flush=True)


if __name__ == "__main__":
    main()
