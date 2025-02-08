import pandas as pd
import ast
import argparse
import re
import os
from typing import Literal

import torch
import transformers
from transformers import BitsAndBytesConfig
from tqdm import tqdm


tqdm.pandas()

PROMPT_TEMPLATE =  """You are given a question, a set of gold-standard reference answers written by experts, and a candidate answer.
Please rate the accuracy of the candidate answer for the question considering the reference answers. Use a scale of 1-3, with 1 indicating an incorrect or irrelevant answer, 2 indicating an ambiguous or incomplete answer, and 3 indicating a correct answer. Give the rationale before rating.

EXAMPLES{EXAMPLES}

Now provide the rating for the following:
Question: {question}
Reference answers: {references}
Candidate answer: {response}
Output:
"""

OPEN_ENDED_EXAMPLES_PROMPT = """
Question: What is the color of the car?
Reference answers: red, scarlet
Candidate answer: pink
Output: The candidate answer is incorrect because the car is 'red' and not 'pink'.
Rating: 1

Question: What is the animal on the left?
Reference answers: elephant, giraffe, giraffe
Candidate answer: giraffe
Output: The candidate answer is correct because most of the reference answers indicate the animal on the left is a giraffe.
Rating: 3

Question: Where are the cookies?
Reference answers: on the left side of the coffee machine
Candidate answer: next to the coffee maker
Output: The candidate answer is incomplete because 'next to' does not specify the side.
Rating: 2

Question: What's the weather like?
Reference answers: bright, bright and sunny, clear, sunny
Candidate answer: cloudy
Output: The candidate answer is incorrect because the weather is 'bright' and 'sunny', not cloudy.
Rating: 1

Question: What are the people in the picture doing?
Reference answers: sitting
Candidate answer: they are resting
Output: The candidate answer is ambiguous because, while it is common that people who are sitting are resting, it is not always the case.
Rating: 2

Question: What color are the base tiles?
Reference answers: beige, brown, tan, ten
Candidate answer: brown
Output: The candidate answer is correct because the reference answers include 'brown' and other similar colors such as 'tan' or 'beige'.
Rating: 3

Question: How many people are in the picture?
Reference answers: four, three, two
Candidate answer: a few
Output: The candidate answer is incomplete because 'a few' is less specific than the numerical reference answers.
Rating: 2

Question: What type of fruit is in the picture?
Reference answers: apple
Candidate answer: fruit
Output: The candidate answer is incorrect because it does not specify the type of fruit.
Rating: 1

Question: What type of sculpture is this?
Reference answers: Horse statue.
Candidate answer: horse
Output: The candidate answer is correct because 'horse' is equivalent to 'horse statue' in this context.
Rating: 3
"""

# for yes / no / and unanswerable
FIXED_EXAMPLES_PROMPT = """
Question: Is the color of the car red?
Reference answers: no
Candidate answer: yes
Output: The candidate answer is incorrect because it's the opposite of the reference answer.
Rating: 1

Question: What is the animal on the left?
Reference answers: Not enough information are depicted in the video to answer this question
Candidate answer: unanswerable
Output: The candidate answer is correct because the video does not provide enough information to answer the question. 
Rating: 3

Question: Are the cookies next to the coffee maker?
Reference answers: Not enough information are depicted in the video to answer this question
Candidate answer: no
Output: The candidate answer is incomplete because 'no' is less specific than the reference answers.
Rating: 2

Question: Do you see a bottle of wine?
Reference answers: yes
Candidate answer: not sure
Output: The candidate answer is incorrect because there is a visible bottle of wine.
Rating: 1

Question: Are the people sitting?
Reference answers: yes
Candidate answer: they are resting
Output: The candidate answer is ambiguous because, while it is common that people who are sitting are resting, it is not always the case.
Rating: 2

Question: Does the weather look cloudy?
Reference answers: no, no
Candidate answer: not really
Output: The candidate answer is correct because 'not really' implies the same as the reference answer 'no'.
Rating: 3

Question: Are there any mugs on the counter top?
Reference answers: yes
Candidate answer: there are two mugs
Output: The candidate answer is incomplete because the number of the mugs cannot be validated based on the candidate answer.
Rating: 2

Question: What type of fruit is in the picture?
Reference answers: Not enough information are depicted in the video to answer this question
Candidate answer: apple
Output: The candidate answer is incorrect because it is not based on the video.
Rating: 1

Question: Is this a bathroom?
Reference answers: yes, yes, no
Candidate answer: yes
Output: The candidate answer is correct because the majority of the reference answers indicate that this is a bathroom.
Rating: 3
"""


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
    "《",
    "》",
    ">",
    "<",
    "=",
    "+",
    "\\",
    "_",
    "-",
    ">",
    "<",
    "@",
    "#",
    "|",
    "%",
    "&",
    "*",
    "`",
    "、",
    "''",
    "'",
    "``",
    ",",
    "?",
    "!",
    ":",
    "。",
]


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


def normalize_answer(answer: str) -> str:
    """Normalize a VQA answer."""
    if answer is None or not isinstance(answer, str):
        answer = "no response"
    answer = answer.lower()
    answer = answer.replace("\n", " ")
    answer = answer.replace("\t", " ")
    answer = answer.strip()
    answer = process_digit_article(process_punctuation(answer))
    return answer


def prepare_ground_truths(ground_truths):
    ground_truths = ast.literal_eval(ground_truths)
    return [normalize_answer(gt) for gt in ground_truths]


class LAVE:
    def __init__(
        self,
        model_id: str,
        load_in_8bit: bool,
        data_file: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        max_new_tokens: int = 256,
    ):
        self.model_id = model_id
        self.load_in_8bit = load_in_8bit
        self.data_file = data_file
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        quantization_config = BitsAndBytesConfig(load_in_8bit=args.load_in_8bit)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=args.model_id,
            model_kwargs={"torch_dtype": torch_dtype, "quantization_config": quantization_config},
            device_map="auto",
        )
        self.pipeline.tokenizer.pad_token_id = self.pipeline.tokenizer.eos_token_id

    def __call__(self, sample: pd.Series):
        messages = self._prepare_input(sample["prompt"], sample["response"], sample["ground_truth"], sample["question_type"])
        outputs = self.pipeline(messages, max_new_tokens=self.max_new_tokens)
        return self.score_sample(outputs)[0]

    def score_sample(self, model_outputs) -> list[float]:
        generated_texts = [output["generated_text"][-1]["content"] for output in model_outputs]
        return [self._parse_score(generated_text) for generated_text in generated_texts]

    def _prepare_input(self, question: str, prediction: str, references: list[str], question_type: Literal["O", "D", "S", "A"]):
        # is any of the refs "yes", "no", "unanswerable"?
        is_closed_form = any([ans in ["yes", "no", "unanswerable"] for ans in references])
        references = ", ".join(references)
        if question_type == "A" or is_closed_form:
            prompt = PROMPT_TEMPLATE.format(
                EXAMPLES=FIXED_EXAMPLES_PROMPT,
                question=question,
                references=references,
                response=prediction,
            )
        else:
            prompt = PROMPT_TEMPLATE.format(
                EXAMPLES=OPEN_ENDED_EXAMPLES_PROMPT,
                question=question,
                references=references,
                response=prediction,
            )
        messages = [
            {"role": "system", "content": "You are a helpful and fair judge."},
            {"role": "user", "content": prompt},
        ]
        return messages

    def _parse_score(self, generated_text: str) -> float:
        generated_text = generated_text.lower().strip()
        try:
            score = float(generated_text.split("rating: ")[-1][0])
        except:
            score = 1
        # normalize the generated text
        return float(score - 1) / 2


def parse_arguments():
    parser = argparse.ArgumentParser(description="Compute the accuracy of the model.")
    parser.add_argument("--data_file", type=str, required=True, help="The results file.")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.3-70B-Instruct", help="The model id.")
    parser.add_argument("--load_in_8bit", action="store_true", help="Whether to load the model in 8-bit.")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="The maximum number of new tokens.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    # Load the data
    data = pd.read_csv(args.data_file, sep="\t", quoting=3, escapechar="\\", quotechar='"')
    data["response"] = data.apply(lambda x: normalize_answer(x["response"]), axis=1)
    data["ground_truth"] = data.apply(lambda x: prepare_ground_truths(x["ground_truth"]), axis=1)
    # Compute the accuracy
    lave_metric = LAVE(args.model_id, args.load_in_8bit, args.data_file, max_new_tokens=args.max_new_tokens)
    data = data.assign(acc=data.progress_apply(lave_metric, axis=1))
    # Save the results
    output_file = os.path.join(os.path.dirname(args.data_file), f"{os.path.basename(args.data_file)}-outputs-acc.csv")
    data.to_csv(output_file, sep="\t", index=False)
    # Compute the average accuracy
    acc = data["acc"].mean()
    print(f"Average accuracy: {acc}")
    # Compute the average accuracy for is_vip_object == True
    acc_vip = data[data["is_vip_object"] == True]["acc"].mean()
    # Compute the average accuracy for is_vip_object == False
    acc_non_vip = data[data["is_vip_object"] == False ]["acc"].mean()
    # Print the results
    print(f"Average accuracy for is_vip_object == True: {acc_vip}")
    print(f"Average accuracy for is_vip_object == False: {acc_non_vip}")
    for question_type in ["O", "D", "S", "A"]:
        print(f"\nQuestion Type = {question_type}")
        # Compute the average accuracy for is_vip_object == True
        acc_vip = data[(data["is_vip_object"] == True) & (data["question_type"] == question_type)]["acc"].mean()
        # Compute the average accuracy for is_vip_object == False
        acc_non_vip = data[(data["is_vip_object"] == False) & (data["question_type"] == question_type)]["acc"].mean()
        # Print the results
        print(f"Average accuracy for is_vip_object == True: {acc_vip}")
        print(f"Average accuracy for is_vip_object == False: {acc_non_vip}")
