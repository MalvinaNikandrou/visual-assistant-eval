import re


def get_n_grams(words: list[str], n: int) -> list[str]:
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


class WordRepetitionFilter:
    def __init__(
        self,
        top_n_grams: tuple[tuple[int, int]] = ((1, 3), (2, 2), (3, 2), (4, 2)),
        dup_n_grams: tuple[tuple[int, float]] = (
            (3, 0.1),
            (4, 0.16),
            (5, 0.15),
            (6, 0.14),
            (7, 0.13),
            (8, 0.12),
            (9, 0.11),
            (10, 0.10),
        ),
    ) -> None:
        """
        Modified from https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/filters/gopher_repetition_filter.py
        """
        self.top_n_grams = top_n_grams
        self.dup_n_grams = dup_n_grams

    def get_max_repeated_n_grams(self, n_grams: list[str]) -> int:
        max_repetitions = 0
        cur_repetitions = 1
        repeated_ngram = n_grams[0]
        for n_gram in n_grams[1:]:
            if n_gram == repeated_ngram:
                cur_repetitions += 1
            else:
                repeated_ngram = n_gram
                max_repetitions = max(max_repetitions, cur_repetitions)
                cur_repetitions = 1
        return max(max_repetitions, cur_repetitions)

    def get_max_repeated_character(self, text: str) -> int:
        res = [len(sub.group()) for sub in re.finditer(r"(.)\1*", text)]
        if not len(res):
            return 0
        return max(res)

    def get_max_repeated_characters(self, text: str) -> bool:
        # find repetitions of two or more consecutive characters, e.g. abab, abcabc etc.
        pattern = "\w"
        for n in range(2, 5):
            pattern = f"{pattern}\w"
            res = [len(sub.group()) for sub in re.finditer(rf"({pattern})\1+", text)]
            if len(res) and max(res) > 2 * n:
                return True
        return False

    def __call__(self, text: str, **kwargs) -> bool:
        """Filter repeated words / ngrams in text.

        Returns True if the text contains repeated words / ngrams, False otherwise.
        """
        words = text.split()

        for n, n_frac in self.top_n_grams:
            n_grams = get_n_grams(words, n)
            if not n_grams:
                continue
            max_repetitions = self.get_max_repeated_n_grams(n_grams)
            if max_repetitions >= n_frac:
                return True

        n_duplicates_char = self.get_max_repeated_character(text)
        for n, n_frac in self.dup_n_grams:
            if n_duplicates_char < n:
                continue
            if n_duplicates_char / len(text) > n_frac:
                return True

        if len(words) == 1:
            return self.get_max_repeated_characters(text)
        return False
