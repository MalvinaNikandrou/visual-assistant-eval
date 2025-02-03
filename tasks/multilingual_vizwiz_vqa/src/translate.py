import argparse
import json
from argparse import Namespace

import torch

from constants import Language, UnanswerableMapping, lang_flores200_codes, SEED
from nllb_translate import NLLBTranslate
from filter import WordRepetitionFilter

torch.manual_seed(SEED)


def get_input_arguments() -> Namespace:
    parser = argparse.ArgumentParser(description="Prepare the machine translated data.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/nllb-200-distilled-1.3B",
        help="The hf model id.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="tasks/multilingual_vizwiz_vqa/data/val_subsample_en.json",
        help="The vizwiz annotations in English.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="Specify which gpu to use",
    )
    args = parser.parse_args()
    return args


def main() -> None:
    args = get_input_arguments()
    with open(args.dataset_dir, "r") as f:
        data = json.load(f)

    questions = [sample["question"] for sample in data]
    combined_qa = []
    sample_ids = []
    answers = []
    for sample_id, sample in enumerate(data):
        for ans in sample["answers"]:
            sample_ids.append(sample_id)
            combined_qa.append(sample["question"] + "\n\n" + ans["answer"])
        answers.extend([ans["answer"] for ans in sample["answers"]])

    batch_size = args.batch_size
    question_translator = NLLBTranslate(
        n_samples=5,
        model_name=args.model_name,
        device=args.gpu,
        torch_dtype=torch.half,
        batch_size=batch_size,
    )
    answer_translator = NLLBTranslate(
        n_samples=3,
        model_name=args.model_name,
        device=args.gpu,
        torch_dtype=torch.half,
        batch_size=batch_size,
    )
    for tgt_lang in lang_flores200_codes:
        assert tgt_lang.name in UnanswerableMapping

    # Filter to catch word / character repetitions
    word_repetition = WordRepetitionFilter()

    # Translate to all languages
    for tgt_lang in lang_flores200_codes:
        if tgt_lang == Language.en:
            continue
        print(f"Loading the {tgt_lang.name} translation pipeline")

        question_translations = question_translator(questions, Language.en, tgt_lang)
        answer_translations = answer_translator(answers, Language.en, tgt_lang)
        annotations = []
        for sample in data:
            new_sample = sample.copy()
            new_sample["question"] = question_translations.pop(0)
            if word_repetition(new_sample["question"]):
                # print a warning
                print(f"Warning: {new_sample['question']} for language {tgt_lang.value} contains repetition")
            for ans in new_sample["answers"]:
                ans["translated_answer"] = answer_translations.pop(0)
                if ans["answer"] == "unanswerable":
                    ans["translated_answer"] = UnanswerableMapping[tgt_lang.name]
                    if word_repetition(new_sample["question"]):
                        # print a warning
                        print(
                            f"Warning: Answer {new_sample['question']} for language {tgt_lang.value} contains repetition"
                        )
            annotations.append(new_sample)

        target_file = f"data/val_subsample_{tgt_lang.name}.json"

        with open(target_file, "w") as out_file:
            json.dump(annotations, out_file)


if __name__ == "__main__":
    main()
