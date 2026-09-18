"""Sequence extraction + edit-distance metrics for OCEL evaluation.

The evaluation compares two ordered activity lists per hidden email, read
straight from the reconstruction's ``predicted.jsonl`` (one record per hidden
email):

  * GROUND TRUTH  — the record's ``true_members``: the real ordered activities
    that were hidden/dropped for this email.
  * PREDICTION    — the record's ``pred``: the LLM's ordered predicted
    activities for the same email.

Activities are compared as opaque tokens (one activity string = one token). All
functions treat single- and multi-span logs identically: a single-span email is
just a sequence of length one.

Two precision/recall/F1 families are reported per pair:

  * MULTISET PRF (``multiset_prf``) — order-agnostic overlap via per-token
    minimum counts (Counter intersection).
  * SEQUENCE PRF (``lcs_prf``) — order-aware overlap via the Longest Common
    Subsequence (ROUGE-L). Since LCS <= multiset overlap, every sequence PRF
    value is <= its multiset counterpart; the gap is an ordering penalty.

A Levenshtein-aligned confusion matrix (``levenshtein_align`` /
``pair_alignment``) counts match / substitution / insertion / deletion at the
token level, so substitutions such as ``A_Validating → W_Validate application``
are visible even when sequence lengths differ.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any


# --------------------------------------------------------------------------- #
# Edit distance
# --------------------------------------------------------------------------- #
def levenshtein(a: list[Any], b: list[Any]) -> int:
    """Token-level Levenshtein edit distance between two sequences.

    Each element is one token; equality is exact. Pure stdlib, O(len(a)*len(b))
    time and O(len(b)) space.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, tok_a in enumerate(a, start=1):
        current = [i]
        for j, tok_b in enumerate(b, start=1):
            cost = 0 if tok_a == tok_b else 1
            current.append(min(
                previous[j] + 1,       # deletion
                current[j - 1] + 1,    # insertion
                previous[j - 1] + cost,  # substitution
            ))
        previous = current
    return previous[-1]


def normalized_distance(a: list[Any], b: list[Any]) -> float:
    """Length-normalized edit distance: ``levenshtein(a, b) / max(len(a), len(b), 1)``.

    Ranges 0.0 (identical) .. 1.0 (fully disjoint); 0.0 when both are empty.
    """
    denom = max(len(a), len(b), 1)
    return levenshtein(a, b) / denom


# --------------------------------------------------------------------------- #
# Alignment-based confusion matrix
# --------------------------------------------------------------------------- #
CONFUSION_INSERT = "__insert__"
CONFUSION_DELETE = "__delete__"


def levenshtein_align(
    true: list[Any], pred: list[Any],
) -> "list[tuple[Any | None, Any | None]]":
    """Token-level Levenshtein alignment of ``true`` vs ``pred``.

    Returns ordered ``(true_tok, pred_tok)`` pairs; ``None`` is a gap
    (deletion if on the true side, insertion if on the pred side). The number
    of non-match pairs equals ``levenshtein(true, pred)``.

    When several ops share the same cost, prefer diagonal (match/substitution)
    over delete over insert, so ``[A]`` vs ``[B]`` is one substitution rather
    than delete-A + insert-B.
    """
    n, m = len(true), len(pred)
    if n == 0 and m == 0:
        return []
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if true[i - 1] == pred[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,        # deletion
                dp[i][j - 1] + 1,        # insertion
                dp[i - 1][j - 1] + cost,  # match / substitution
            )

    i, j = n, m
    rev: list[tuple[Any | None, Any | None]] = []
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if true[i - 1] == pred[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                rev.append((true[i - 1], pred[j - 1]))
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            rev.append((true[i - 1], None))
            i -= 1
            continue
        rev.append((None, pred[j - 1]))
        j -= 1
    rev.reverse()
    return rev


def pair_alignment(
    true: list[str], pred: list[str],
) -> "tuple[dict[str, int], list[list[str]], Counter]":
    """Per-pair alignment ops, substitution pairs, and confusion cells.

    ``ops`` counts ``match`` / ``sub`` / ``ins`` / ``del``. ``subs`` is
    ``[[true_label, pred_label], ...]`` for substitutions. The ``Counter``
    keys are ``(row, col)`` with sentinels ``CONFUSION_INSERT`` /
    ``CONFUSION_DELETE`` for gaps.
    """
    ops = {"match": 0, "sub": 0, "ins": 0, "del": 0}
    subs: list[list[str]] = []
    counts: Counter = Counter()
    for t, p in levenshtein_align(true, pred):
        if t is None:
            ops["ins"] += 1
            counts[(CONFUSION_INSERT, p)] += 1
        elif p is None:
            ops["del"] += 1
            counts[(t, CONFUSION_DELETE)] += 1
        elif t == p:
            ops["match"] += 1
            counts[(t, p)] += 1
        else:
            ops["sub"] += 1
            subs.append([t, p])
            counts[(t, p)] += 1
    return ops, subs, counts


def confusion_report(counts: Counter) -> dict:
    """JSON-serializable confusion matrix from a ``(row, col) -> count`` Counter.

    ``labels`` are observed activity names (sorted) then the two sentinels.
    ``matrix[i][j]`` is the count for ``labels[i]`` (true) × ``labels[j]``
    (pred). ``substitutions`` lists off-diagonal non-sentinel pairs as
    ``[true, pred, count]`` sorted by count descending.
    """
    activities: set[str] = set()
    for row, col in counts:
        if row != CONFUSION_INSERT:
            activities.add(row)
        if col != CONFUSION_DELETE:
            activities.add(col)
    labels = sorted(activities) + [CONFUSION_INSERT, CONFUSION_DELETE]
    index = {lab: i for i, lab in enumerate(labels)}
    n = len(labels)
    matrix = [[0] * n for _ in range(n)]
    n_match = n_sub = n_ins = n_del = 0
    substitutions: list[list] = []
    for (row, col), c in counts.items():
        matrix[index[row]][index[col]] = c
        if row == CONFUSION_INSERT:
            n_ins += c
        elif col == CONFUSION_DELETE:
            n_del += c
        elif row == col:
            n_match += c
        else:
            n_sub += c
            substitutions.append([row, col, c])
    substitutions.sort(key=lambda x: (-x[2], x[0], x[1]))
    return {
        "labels": labels,
        "matrix": matrix,
        "n_aligned": n_match + n_sub + n_ins + n_del,
        "n_match": n_match,
        "n_sub": n_sub,
        "n_ins": n_ins,
        "n_del": n_del,
        "substitutions": substitutions,
    }


# --------------------------------------------------------------------------- #
# Set / multiset overlap (order-agnostic, partial credit)
# --------------------------------------------------------------------------- #
def multiset_prf(true: list[Any], pred: list[Any]) -> "dict[str, float]":
    """Precision / recall / F1 over activity MULTISETS (order ignored, counts kept).

    Overlap is the sum of per-token minimum counts (``Counter`` intersection),
    so a correct-plus-extra prediction earns partial credit instead of 0. Both
    empty -> all 1.0.
    """
    ct, cp = Counter(true), Counter(pred)
    overlap = sum((ct & cp).values())
    precision = overlap / len(pred) if pred else (1.0 if not true else 0.0)
    recall = overlap / len(true) if true else (1.0 if not pred else 0.0)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def jaccard(true: list[Any], pred: list[Any]) -> float:
    """Set overlap ``|A ∩ B| / |A ∪ B|`` (order and counts ignored).

    1.0 when both are empty.
    """
    sa, sb = set(true), set(pred)
    union = sa | sb
    if not union:
        return 1.0
    return len(sa & sb) / len(union)


def dice_coefficient(true: list[Any], pred: list[Any]) -> float:
    """Sørensen–Dice over activity TYPES: ``2|A ∩ B| / (|A| + |B|)``.

    Same information as Jaccard but weights the intersection more heavily
    (``dice = 2J / (1 + J)``); order and counts ignored. 1.0 when both empty.
    """
    sa, sb = set(true), set(pred)
    denom = len(sa) + len(sb)
    if denom == 0:
        return 1.0
    return 2 * len(sa & sb) / denom


def cosine_similarity(true: list[Any], pred: list[Any]) -> float:
    """Cosine similarity of activity COUNT vectors (bag of activities).

    Treats each sequence as a vector of per-activity counts and measures the
    angle between them, so it rewards matching *proportions* of activities
    regardless of order. 1.0 when both empty; 0.0 when exactly one is empty.
    """
    ct, cp = Counter(true), Counter(pred)
    if not ct and not cp:
        return 1.0
    dot = sum(ct[k] * cp[k] for k in ct.keys() & cp.keys())
    norm_t = math.sqrt(sum(v * v for v in ct.values()))
    norm_p = math.sqrt(sum(v * v for v in cp.values()))
    if norm_t == 0 or norm_p == 0:
        return 0.0
    return dot / (norm_t * norm_p)


# --------------------------------------------------------------------------- #
# Sequence-aware overlap / ordering (order matters)
# --------------------------------------------------------------------------- #
def lcs_length(a: list[Any], b: list[Any]) -> int:
    """Length of the Longest Common Subsequence of ``a`` and ``b``.

    A subsequence keeps relative order but allows gaps, so LCS counts how many
    activities appear in the same order in both traces. O(len(a)*len(b)) time,
    O(len(b)) space.
    """
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for tok_a in a:
        current = [0]
        for j, tok_b in enumerate(b, start=1):
            if tok_a == tok_b:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[j - 1]))
        previous = current
    return previous[-1]


def lcs_ratio(true: list[Any], pred: list[Any]) -> float:
    """LCS length normalized by the longer sequence: ``LCS / max(len)``.

    Order-aware overlap in [0, 1]: 1.0 means one is an in-order subsequence of
    the other of equal length (i.e. identical). 1.0 when both empty.
    """
    if not true and not pred:
        return 1.0
    return lcs_length(true, pred) / max(len(true), len(pred), 1)


def lcs_prf(true: list[Any], pred: list[Any]) -> "dict[str, float]":
    """Order-aware precision / recall / F1 via the LCS (ROUGE-L style).

    Overlap is the Longest Common Subsequence length, so unlike ``multiset_prf``
    (which counts unordered per-token minima) this only credits activities that
    appear in the SAME relative order in both sequences::

        precision = LCS(true, pred) / len(pred)
        recall    = LCS(true, pred) / len(true)
        f1        = harmonic mean

    Because LCS respects order, ``LCS <= multiset overlap`` always, hence every
    sequence PRF value is <= its multiset PRF counterpart; they coincide exactly
    when the shared activities are already correctly ordered. Both empty -> all
    1.0; exactly one empty -> all 0.0.
    """
    if not true and not pred:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    overlap = lcs_length(true, pred)
    precision = overlap / len(pred) if pred else 0.0
    recall = overlap / len(true) if true else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def kendall_tau(true: list[Any], pred: list[Any]) -> float:
    """Kendall rank correlation of activities present in BOTH sequences.

    Uses each activity type's FIRST-occurrence index in each sequence as its
    rank, then measures how many ordered pairs agree (concordant) vs disagree
    (discordant): ``tau = (C - D) / (n(n-1)/2)``. Range -1.0 (reversed order)
    .. 1.0 (same order). Returns 1.0 when fewer than two shared activity types
    (no pair can disagree). Ignores counts and non-shared activities.
    """
    first_t: dict[Any, int] = {}
    for i, tok in enumerate(true):
        first_t.setdefault(tok, i)
    first_p: dict[Any, int] = {}
    for i, tok in enumerate(pred):
        first_p.setdefault(tok, i)

    shared = [tok for tok in first_t if tok in first_p]
    n = len(shared)
    if n < 2:
        return 1.0

    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            x, y = shared[i], shared[j]
            sign = (first_t[x] - first_t[y]) * (first_p[x] - first_p[y])
            if sign > 0:
                concordant += 1
            elif sign < 0:
                discordant += 1
    return (concordant - discordant) / (n * (n - 1) / 2)


# --------------------------------------------------------------------------- #
# Per-pair metrics + error taxonomy (experiment scoring)
# --------------------------------------------------------------------------- #
def pair_metrics(true: list[str], pred: list[str]) -> "dict[str, Any]":
    """Standard sequence metrics for one (true, pred) pair (per-email view)."""
    ms = multiset_prf(true, pred)
    seq = lcs_prf(true, pred)
    return {
        "true_length": len(true),
        "pred_length": len(pred),
        "length_bias": len(pred) - len(true),
        "edit_distance": levenshtein(true, pred),
        "normalized_edit_distance": normalized_distance(true, pred),
        "multiset_f1": ms["f1"],
        "multiset_precision": ms["precision"],
        "multiset_recall": ms["recall"],
        "seq_f1": seq["f1"],
        "seq_precision": seq["precision"],
        "seq_recall": seq["recall"],
        "jaccard": jaccard(true, pred),
        "lcs_ratio": lcs_ratio(true, pred),
        "dice": dice_coefficient(true, pred),
        "cosine": cosine_similarity(true, pred),
        "exact_match": true == pred,
    }


def fp_fn_counts(true: list[str], pred: list[str]) -> "tuple[dict[str, int], dict[str, int]]":
    """Per-activity false positives / false negatives between two multisets.

    ``fp[a]`` = how many extra ``a`` the prediction has (over-count / invented);
    ``fn[a]`` = how many ``a`` the prediction is missing (under-count / dropped).
    """
    ct, cp = Counter(true), Counter(pred)
    fp = {a: c for a, c in (cp - ct).items()}
    fn = {a: c for a, c in (ct - cp).items()}
    return fp, fn


def categorize_pair(true: list[str], pred: list[str]) -> str:
    """Label the discrepancy between a true and predicted activity list.

    Mirrors ``analysis/archive/analyze_pe_runs.py::categorize`` so the reports share one
    taxonomy: exact / multiplicity(over|under) / ordering / over-generation /
    under-generation / substitution/mixed.
    """
    if true == pred:
        return "exact"
    ct, cp = Counter(true), Counter(pred)
    only_true = set(ct) - set(cp)
    only_pred = set(cp) - set(ct)
    if ct == cp:
        return "ordering"
    if not only_true and not only_pred:
        return "multiplicity(over)" if len(pred) > len(true) else "multiplicity(under)"
    if only_pred and not only_true:
        return "over-generation"
    if only_true and not only_pred:
        return "under-generation"
    return "substitution/mixed"
