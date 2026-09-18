"""Experiment scoring + aggregation (Pass B - score), embedded here.

Reads the reconstruction component's ``predicted.jsonl`` (each record carries
``true_members`` and the LLM ``pred``) and scores every hidden email
INDEPENDENTLY with the shared pure-stdlib metrics. Results are written to
``summary.json`` + ``.md`` and ``scored_rows.jsonl``: an overall roll-up plus
breakdowns by the dropped pattern's size ``n`` and the individual dropped unit,
and a Levenshtein-aligned true×pred confusion matrix (with insert/delete).
No LLM / Azure / network.
"""
from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from src.metrics import (
    CONFUSION_DELETE,
    CONFUSION_INSERT,
    categorize_pair,
    confusion_report,
    fp_fn_counts,
    pair_alignment,
    pair_metrics,
)

# Numeric metrics averaged across a bucket of (true, pred) pairs.
_MEAN_KEYS = [
    "normalized_edit_distance",
    "multiset_f1",
    "multiset_precision",
    "multiset_recall",
    "seq_f1",
    "seq_precision",
    "seq_recall",
    "jaccard",
    "lcs_ratio",
    "length_bias",
]


def _summarize(pairs: list[dict]) -> dict:
    """Mean of the numeric metrics + exact-match rate + category mix for a bucket."""
    if not pairs:
        return {"n": 0}
    out: dict = {"n": len(pairs)}
    for key in _MEAN_KEYS:
        vals = [p[key] for p in pairs]
        out[key] = round(statistics.fmean(vals), 4)
        out[f"{key}_std"] = round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0
    out["exact_rate"] = round(
        statistics.fmean(1.0 if p["exact_match"] else 0.0 for p in pairs), 4
    )
    cats: Counter = Counter(p["category"] for p in pairs)
    out["categories"] = dict(cats.most_common())
    return out


def run_aggregate(
    inp_path: str | Path,
    out_dir: str | Path,
) -> tuple[int, Path]:
    """Score + aggregate the predicted records, writing the summary files.

    Returns ``(n_scored, summary_json_path)``.
    """
    records = [
        json.loads(line)
        for line in Path(inp_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"loaded {len(records)} record(s) from {inp_path}", flush=True)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "summary.json"
    out_md = out_dir / "summary.md"

    overall: list[dict] = []
    by_size: dict = defaultdict(list)   # size n -> pairs
    by_unit: dict = defaultdict(list)   # unit_id -> pairs
    unit_meta: dict = {}
    scored_rows: list[dict] = []
    confusion_overall: Counter = Counter()
    confusion_by_size: dict = defaultdict(Counter)

    for r in records:
        if not r.get("email"):
            continue  # non-email placeholder record: nothing to score
        true = r.get("true_members", [])
        pred = r.get("pred", [])
        m = pair_metrics(true, pred)
        m["category"] = categorize_pair(true, pred)
        fp, fn = fp_fn_counts(true, pred)
        ops, subs, conf_cells = pair_alignment(true, pred)

        size = r.get("gram_size", r.get("size")) or len(true)
        unit_id = r.get("unit_id", "")
        overall.append(m)
        by_size[size].append(m)
        by_unit[unit_id].append(m)
        unit_meta.setdefault(unit_id, {"size": size})
        confusion_overall.update(conf_cells)
        confusion_by_size[size].update(conf_cells)
        scored_rows.append({
            "unit_id": unit_id, "size": size,
            "case_id": r.get("case_id", ""), "true": true, "pred": pred,
            "confidence": r.get("confidence", 0.0),
            "category": m["category"], "fp": fp, "fn": fn,
            "ops": ops, "subs": subs,
        })

    size_keys = sorted(by_size, key=lambda k: (k is None, k))
    confusion = confusion_report(confusion_overall)
    confusion["by_size"] = {
        str(k): confusion_report(confusion_by_size[k]) for k in size_keys
    }
    summary = {
        "config": {"n_records": len(records), "n_scored": len(scored_rows)},
        "overall": _summarize(overall),
        "by_size": {str(k): _summarize(by_size[k]) for k in size_keys},
        "by_unit": {uid: {**unit_meta[uid], **_summarize(pairs)}
                    for uid, pairs in by_unit.items()},
        "confusion": confusion,
    }

    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    (out_dir / "scored_rows.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in scored_rows),
        encoding="utf-8",
    )
    _write_md(out_md, summary)
    print(f"Pass B score complete: scored {len(scored_rows)} email(s) "
          f"-> {out_json}", flush=True)
    return len(scored_rows), out_json


def _fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, float) else str(v)


# (column header, summary key) for the per-bucket metric tables, in order.
_METRIC_FIELDS = [
    ("exact", "exact_rate"),
    ("ms_f1", "multiset_f1"),
    ("ms_prec", "multiset_precision"),
    ("ms_rec", "multiset_recall"),
    ("seq_f1", "seq_f1"),
    ("seq_prec", "seq_precision"),
    ("seq_rec", "seq_recall"),
    ("norm_edit", "normalized_edit_distance"),
    ("jaccard", "jaccard"),
    ("len_bias", "length_bias"),
]
_METRIC_COLS = " | ".join(h for h, _ in _METRIC_FIELDS)
_METRIC_SEP = "".join("|" + "-" * max(len(h), 3) for h, _ in _METRIC_FIELDS) + "|"


def _fmt_ms(s: dict, key: str) -> str:
    """``mean ± std`` for a numeric metric; ``exact_rate`` is a plain proportion."""
    if key == "exact_rate":
        return _fmt(s.get(key))
    mean = s.get(key)
    std = s.get(f"{key}_std")
    if not isinstance(mean, float):
        return _fmt(mean)
    std_val = std if isinstance(std, float) else 0.0
    return f"{mean:.3f} ± {std_val:.3f}"


def _metric_cells(s: dict) -> str:
    return " | ".join(_fmt_ms(s, key) for _, key in _METRIC_FIELDS)


def _rate(n: int, denom: int) -> str:
    return f"{n / denom:.3f}" if denom else "0.000"


def _confusion_ops_table(conf: dict) -> list[str]:
    """Match / sub / ins / del counts with rates over aligned tokens."""
    n = conf.get("n_aligned", 0)
    return [
        f"Aligned tokens: {n}. Rates are over aligned tokens.",
        "",
        "| match | sub | ins | del |",
        "|------:|----:|----:|----:|",
        f"| {conf.get('n_match', 0)} ({_rate(conf.get('n_match', 0), n)}) "
        f"| {conf.get('n_sub', 0)} ({_rate(conf.get('n_sub', 0), n)}) "
        f"| {conf.get('n_ins', 0)} ({_rate(conf.get('n_ins', 0), n)}) "
        f"| {conf.get('n_del', 0)} ({_rate(conf.get('n_del', 0), n)}) |",
    ]


def _confusion_subs_table(conf: dict) -> list[str]:
    subs = conf.get("substitutions") or []
    if not subs:
        return ["No substitutions."]
    lines = [
        "Top substitutions (true → pred):",
        "",
        "| true | pred | count |",
        "|------|------|------:|",
    ]
    for true_lab, pred_lab, count in subs:
        lines.append(f"| {true_lab} | {pred_lab} | {count} |")
    return lines


def _confusion_sparse_matrix(conf: dict) -> list[str]:
    """Sparse count matrix: drop all-zero activity labels; keep insert row + delete col."""
    labels = conf.get("labels") or []
    matrix = conf.get("matrix") or []
    if not labels or not matrix:
        return ["No aligned tokens."]
    idx = {lab: i for i, lab in enumerate(labels)}
    activities = [l for l in labels if l not in (CONFUSION_INSERT, CONFUSION_DELETE)]
    keep = []
    n = len(labels)
    for a in activities:
        i = idx[a]
        if any(matrix[i][j] for j in range(n)) or any(matrix[r][i] for r in range(n)):
            keep.append(a)
    row_labels = keep + [CONFUSION_INSERT]
    col_labels = keep + [CONFUSION_DELETE]
    header = "| true \\ pred | " + " | ".join(col_labels) + " |"
    sep = "|--------------|" + "|".join("-" * max(len(c), 3) for c in col_labels) + "|"
    lines = [header, sep]
    for row in row_labels:
        ri = idx[row]
        cells = " | ".join(str(matrix[ri][idx[col]]) for col in col_labels)
        lines.append(f"| {row} | {cells} |")
    return lines


def _has_non_match(conf: dict) -> bool:
    return bool(
        conf.get("n_sub") or conf.get("n_ins") or conf.get("n_del")
    )


def _write_md(path: Path, summary: dict) -> None:
    cfg = summary["config"]
    lines = [
        "# Reconstruction accuracy",
        "",
        f"- Records: {cfg['n_records']}  |  Scored emails: {cfg['n_scored']}",
        "",
        "Higher exact/F1 and lower normalized edit distance = better. `ms_*` = "
        "multiset PRF (order-agnostic); `seq_*` = sequence PRF via LCS/ROUGE-L "
        "(order-aware, always <= the multiset value).",
        "",
        "## Overall",
        "",
        f"| n | {_METRIC_COLS} |",
        f"|---{_METRIC_SEP}",
    ]
    ov = summary["overall"]
    if ov.get("n"):
        lines.append(f"| {ov['n']} | {_metric_cells(ov)} |")

    lines += ["", "## By size n", "",
              f"| value | n | {_METRIC_COLS} |",
              f"|-------|---{_METRIC_SEP}"]
    for val, s in summary["by_size"].items():
        if s.get("n"):
            lines.append(f"| {val} | {s['n']} | {_metric_cells(s)} |")

    lines += ["", "## By dropped unit", "",
              "| unit | size | n | exact | multiset_f1 | norm_edit |",
              "|------|------|---|-------|-------------|-----------|"]
    for uid, s in summary["by_unit"].items():
        if not s.get("n"):
            continue
        lines.append(
            f"| {uid} | {s.get('size')} | {s['n']} | "
            f"{_fmt(s['exact_rate'])} | {_fmt_ms(s, 'multiset_f1')} | "
            f"{_fmt_ms(s, 'normalized_edit_distance')} |"
        )

    conf = summary.get("confusion") or {}
    lines += [
        "",
        "## Confusion",
        "",
        "Token-level confusion from a Levenshtein alignment of `true_members` vs "
        "`pred`. Rows are true activities, columns are predicted activities. "
        "`__insert__` is a predicted token with no true counterpart; `__delete__` "
        "is a true token with no predicted counterpart. Tie-break prefers "
        "substitution over insert+delete, so `[A]` vs `[B]` counts as `A → B`.",
        "",
    ]
    lines += _confusion_ops_table(conf)
    lines += ["", *_confusion_subs_table(conf), ""]
    lines += ["Sparse count matrix (all-zero activity labels dropped):", ""]
    lines += _confusion_sparse_matrix(conf)

    by_size_conf = conf.get("by_size") or {}
    size_sections = [
        (val, by_size_conf[val])
        for val in summary["by_size"]
        if val in by_size_conf and _has_non_match(by_size_conf[val])
    ]
    if size_sections:
        lines += ["", "### By size n"]
        for val, sc in size_sections:
            lines += ["", f"#### n = {val}", ""]
            lines += _confusion_ops_table(sc)
            lines += ["", *_confusion_subs_table(sc)]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
