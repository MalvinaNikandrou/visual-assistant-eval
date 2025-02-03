import re
import string
from collections import Counter, defaultdict
from typing import Any


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join(["‘", "’", "´", "`"]))
        return "".join(ch if ch not in exclude else " " for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace("_", " ")

    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()


def round_to_base(num, base=5):
    return base * round(num / base)


class SquadMetric:
    def compute(
        self,
        predictions: list[str],
        ground_truths: list[list[str]],
        context_sizes: list[int],
        bin_size: int = 10,
    ) -> dict[str, Any]:
        total_em = 0
        total_f1 = 0
        em_per_bin = defaultdict(list)
        f1_per_bin = defaultdict(list)
        for pred, gt, context_size in zip(predictions, ground_truths, context_sizes):
            # em = exact_match_score(pred, gt)
            em = metric_max_over_ground_truths(exact_match_score, pred, gt)
            # f1 = f1_score(pred, gt)
            f1 = metric_max_over_ground_truths(f1_score, pred, gt)
            total_em += em
            total_f1 += f1
            bin_idx = round_to_base(context_size, bin_size)
            em_per_bin[bin_idx].append(em)
            f1_per_bin[bin_idx].append(f1)

        return {
            "em": round(total_em / len(predictions) * 100, 2),
            "f1": round(total_f1 / len(predictions) * 100, 2),
            "em_per_bin": {
                bin_idx: round(sum(em_per_bin[bin_idx]) / len(em_per_bin[bin_idx]) * 100, 2) for bin_idx in em_per_bin
            },
            "f1_per_bin": {
                bin_idx: round(sum(f1_per_bin[bin_idx]) / len(f1_per_bin[bin_idx]) * 100, 2) for bin_idx in f1_per_bin
            },
        }


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: list[str]) -> float:
    scores_for_ground_truths = []
    if not ground_truths:
        ground_truths = ["unanswerable"]

    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)
