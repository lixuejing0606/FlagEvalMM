"""OmniDocBench evaluation for FlagEvalMM.

Metrics (following the official OmniDocBench evaluation):
  - Normalized Edit Distance (primary, via Levenshtein)
  - Overall score = (1 - edit_distance) * 100

The official OmniDocBench benchmark also computes TEDS (tables) and CDM
(formulas) separately, but those require heavy system dependencies (TexLive,
ImageMagick, apted, etc.).  This evaluator uses the end-to-end edit distance
on the full Markdown output, which is what the Overall score is derived from
in the official leaderboard for the VLM end-to-end task.
"""

from typing import Dict, List
from collections import defaultdict

try:
    import Levenshtein
except ImportError:
    Levenshtein = None

import re
import unicodedata


# ---------------------------------------------------------------------------
# Text normalisation (applied to both GT and prediction before comparison)
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Light normalisation to reduce superficial mismatches."""
    text = unicodedata.normalize("NFKC", text)
    # Collapse consecutive whitespace / blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = text.strip()
    return text


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _edit_distance_score(gt: str, pred: str) -> float:
    """Return normalised edit distance in [0, 1].  0 = perfect match."""
    if not gt and not pred:
        return 0.0
    gt_n = _normalize(gt)
    pred_n = _normalize(pred)
    upper = max(len(gt_n), len(pred_n))
    if upper == 0:
        return 0.0
    if Levenshtein is not None:
        dist = Levenshtein.distance(pred_n, gt_n)
    else:
        # Fallback: simple DP (slow but correct)
        dist = _levenshtein_dp(pred_n, gt_n)
    return dist / upper


def _levenshtein_dp(s: str, t: str) -> int:
    """Pure-Python Levenshtein distance (fallback when the C extension is
    unavailable)."""
    m, n = len(s), len(t)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if s[i - 1] == t[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[n]


# ---------------------------------------------------------------------------
# Main entry point called by BaseEvaluator
# ---------------------------------------------------------------------------

def get_result(annotations: Dict, predictions: List[Dict]) -> Dict:
    """Compute OmniDocBench evaluation metrics.

    Parameters
    ----------
    annotations : dict
        {question_id: annotation_dict} from data.json
    predictions : list[dict]
        Each element has at least ``question_id`` and ``answer``.

    Returns
    -------
    dict
        Results dictionary with ``accuracy`` (= (1 - edit_dist) * 100),
        per-subtask scores, and raw ``edit_distance``.
    """
    if not predictions:
        return {"accuracy": 0.0, "edit_distance": 1.0}

    edit_scores = []  # list of (normalised_edit_dist, upper_len)
    sub_task_scores = defaultdict(list)
    language_scores = defaultdict(list)

    for pred in predictions:
        qid = str(pred["question_id"])
        gt = annotations.get(qid, {})
        gt_answer = gt.get("answer", "")
        pred_answer = pred.get("answer", "")

        ed = _edit_distance_score(gt_answer, pred_answer)
        gt_n = _normalize(gt_answer)
        pred_n = _normalize(pred_answer)
        upper = max(len(gt_n), len(pred_n), 1)

        edit_scores.append((ed, upper))

        pred["edit_distance"] = round(ed, 4)
        pred["score"] = round((1 - ed) * 100, 2)
        pred["label"] = gt_answer[:200] + "..." if len(gt_answer) > 200 else gt_answer

        sub_task = gt.get("sub_task", "unknown")
        sub_task_scores[sub_task].append((ed, upper))
        pred["sub_task"] = sub_task

        lang = gt.get("language", "unknown")
        language_scores[lang].append((ed, upper))

    # Weighted average edit distance (weighted by max string length, same as
    # the official OmniDocBench page-level then global average).
    total_dist = sum(ed * ul for ed, ul in edit_scores)
    total_upper = sum(ul for _, ul in edit_scores)
    avg_edit = total_dist / total_upper if total_upper > 0 else 1.0

    results = {
        "accuracy": round((1 - avg_edit) * 100, 2),
        "edit_distance": round(avg_edit, 4),
        "num_samples": len(predictions),
    }

    # Per sub_task (data_source) breakdown
    for st, scores in sorted(sub_task_scores.items()):
        td = sum(ed * ul for ed, ul in scores)
        tu = sum(ul for _, ul in scores)
        st_edit = td / tu if tu > 0 else 1.0
        results[f"sub_task/{st}"] = round((1 - st_edit) * 100, 2)

    # Per language breakdown
    for lang, scores in sorted(language_scores.items()):
        td = sum(ed * ul for ed, ul in scores)
        tu = sum(ul for _, ul in scores)
        lang_edit = td / tu if tu > 0 else 1.0
        results[f"language/{lang}"] = round((1 - lang_edit) * 100, 2)

    return results
