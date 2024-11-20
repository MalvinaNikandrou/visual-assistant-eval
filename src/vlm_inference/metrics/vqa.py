"""Normalize VQA-v2 answers.

From the official evaluation code at https://github.com/GT-Vision-
Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py.
"""
import re
from collections import Counter
from typing import Any, Union

import torch
from overrides import overrides
from torchmetrics import Metric



contractions = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hed've": "he'd've",
    "he'dve": "he'd've",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "Id've": "I'd've",
    "I'dve": "I'd've",
    "Im": "I'm",
    "Ive": "I've",
    "isnt": "isn't",
    "itd": "it'd",
    "itd've": "it'd've",
    "it'dve": "it'd've",
    "itll": "it'll",
    "let's": "let's",
    "maam": "ma'am",
    "mightnt": "mightn't",
    "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "notve": "not've",
    "oclock": "o'clock",
    "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at",
    "shant": "shan't",
    "shed've": "she'd've",
    "she'dve": "she'd've",
    "she's": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll",
    "somebodys": "somebody's",
    "someoned": "someone'd",
    "someoned've": "someone'd've",
    "someone'dve": "someone'd've",
    "someonell": "someone'll",
    "someones": "someone's",
    "somethingd": "something'd",
    "somethingd've": "something'd've",
    "something'dve": "something'd've",
    "somethingll": "something'll",
    "thats": "that's",
    "thered": "there'd",
    "thered've": "there'd've",
    "there'dve": "there'd've",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyd've": "they'd've",
    "they'dve": "they'd've",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "twas": "'twas",
    "wasnt": "wasn't",
    "wed've": "we'd've",
    "we'dve": "we'd've",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whod've": "who'd've",
    "who'dve": "who'd've",
    "wholl": "who'll",
    "whos": "who's",
    "whove": "who've",
    "whyll": "why'll",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've",
    "yall": "y'all",
    "yall'll": "y'all'll",
    "y'allll": "y'all'll",
    "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've",
    "youd": "you'd",
    "youd've": "you'd've",
    "you'dve": "you'd've",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}
digit_map = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
articles = ["a", "an", "the"]


period_strip = re.compile(r"(?!<=\d)(\.)(?!\d)")
comma_strip = re.compile(r"(\d)(\,)(\d)")
punctuations = [
    ";",
    "/",
    "[",
    "]",
    '"',
    "{",
    "}",
    "(",
    ")",
    "=",
    "+",
    "\\",
    "_",
    "-",
    ">",
    "<",
    "@",
    "`",
    ",",
    "?",
    "!",
]

def split_at_first_capital_after_whitespace(s):
    # Pattern: lookbehind for whitespace followed by a capital letter
    pattern = r'(?<=\s)(?=[A-Z])'
    parts = re.split(pattern, s, maxsplit=1)[0]
    parts = parts.split(",")[0]
    return parts.strip()


def normalize_answer(answer: str) -> str:
    """Normalize a VQA answer."""
    answer = answer.lower()
    answer = answer.replace("\n", " ")
    answer = answer.replace("\t", " ")
    answer = answer.strip()
    answer = process_digit_article(process_punctuation(answer))
    if answer.startswith("unanswerable"):
        answer = "unanswerable"
    return answer


def process_punctuation(in_text: str) -> str:
    """Process the answer punctuation."""
    out_text = in_text
    for punct in punctuations:
        punct_cond1 = f"{punct} " in in_text or f" {punct}" in in_text
        punct_cond2 = re.search(comma_strip, in_text) is not None
        if punct_cond1 or punct_cond2:
            out_text = out_text.replace(punct, "")
        else:
            out_text = out_text.replace(punct, " ")
    out_text = period_strip.sub("", out_text, re.UNICODE)
    return out_text


def process_digit_article(in_text: str) -> str:
    """Preprocess digits and articles."""
    out_text = []
    for word in in_text.lower().split():
        word = digit_map.setdefault(word, word)
        if word not in articles:
            out_text.append(word)

    for word_id, word in enumerate(out_text):  # noqa: WPS440
        out_text[word_id] = contractions.get(word, word)
    return " ".join(out_text)


def vqa_v2_score(count: int) -> float:
    """VQA-v2 includes 10 answers for each question.

    Scores are assigned as follows:
    - 0.3 if the answer appears once
    - 0.6 if the answer appears twice
    - 0.9 if the answer appears three times
    - 1.0 if the answer appears more than three times
    """
    return min(1.0, round(0.3 * count, 1))  # noqa: WPS432


class VQAv2Accuracy(Metric):
    """VQAv2 accuracy."""

    def __init__(self, dist_sync_on_step: bool = True) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state(
            "accuracy", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    @overrides(check_signature=False)
    def update(self, predicted_answers: list[str], ground_truth_batch: list[list[str]]) -> None:
        """Update loss sum and number of task samples."""
        for predicted_answer, ground_truth_answers in zip(predicted_answers, ground_truth_batch):
            if isinstance(predicted_answer, dict):
                predicted_answer = predicted_answer["answer"]
            if predicted_answer is None:
                predicted_answer = ""
            predicted_answer = normalize_answer(predicted_answer)
            ground_truth_answers = [normalize_answer(answer) for answer in ground_truth_answers]
            ground_truth_counts = Counter(ground_truth_answers)
            self.accuracy += torch.tensor(
                vqa_v2_score(ground_truth_counts.get(predicted_answer, 0))
            )

        self.total += torch.tensor(len(ground_truth_batch))

    def compute(self) -> dict[str, Union[float, int]]:
        """Compute the total task loss."""
        accuracy = self.accuracy.float() / self.total  # type: ignore[operator]
        return {"overall": accuracy.item()}