"""
Script to subsample VizWiz VQA questions. 
The goal is to get a representative subset of N questions.
"""

import json
import random
from collections import Counter

random.seed(42)

INPUT_FILE = "data/val.json"
OUTPUT_FILE = "tasks/multilingual_vizwiz_vqa/val_subsample_en.json"
NUM_SAMPLES = 500

target_answer_type_counts = {"other": 283, "unanswerable": 150, "yes/no": 22, "number": 45}


class VizWizSubsampler:

    def __init__(self, input_file, output_file, target_answer_type_counts):
        self.target_answer_type_counts = (
            target_answer_type_counts  # {'other': 283, 'unanswerable': 150, 'yes/no': 22, 'number': 45}
        )
        self.data = self.load_data(input_file)
        print(f"Loaded {len(self.data)} questions")
        self.self.question_id_to_sample = {sample["question_id"]: sample for sample in self.data}
        self.question_id_to_answer = self.get_most_common_answer()
        self.output_file = output_file

    def load_data(self, input_file):
        with open(input_file, "r") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} questions")
        # keep samples that have at least 5 "answer_confidence" == "yes" answers
        data = [sample for sample in data if sum([ans["answer_confidence"] == "yes" for ans in sample["answers"]]) >= 5]
        print(f"Kept {len(data)} questions")
        # add a question id to each sample
        for idx, sample in enumerate(data):
            sample["question_id"] = f"{sample['image']}_{idx}"
        return data

    def get_most_common_answer(self):
        self.question_id_to_sample = {sample["question_id"]: sample for sample in self.data}
        #  Get the most common answer per question
        question_id_to_answer = {}
        for sample in self.data:
            answers = [ans["answer"] for ans in sample["answers"]]
            most_common_answer = Counter(answers).most_common(1)[0][0]
            question_id_to_answer[sample["question_id"]] = most_common_answer
        return question_id_to_answer

    def subsample(self):
        sampled_data = []
        # Subsample per answer type
        for answer_type, target_count in target_answer_type_counts.items():
            question_ids = [
                qid for qid, sample in self.question_id_to_sample.items() if sample["answer_type"] == answer_type
            ]
            print(f"Found {len(question_ids)} {answer_type} questions")
            if len(question_ids) < target_count:
                print(f"Warning: {len(question_ids)} {answer_type} questions is less than {target_count}")
            if answer_type == "number":
                # keep all the data
                sampled_data.extend(self._subsample_number_data(question_ids, target_count))
            elif answer_type == "yes/no":
                sampled_data.extend(self._subsample_yes_no_data(question_ids, target_count))
            elif answer_type == "unanswerable":
                sampled_data.extend(self._subsample_unanswerable_data(question_ids, target_count))
            else:
                sampled_data.extend(self._subsample_other_data(question_ids, target_count))

        print(f"Sampled {len(sampled_data)} questions")
        # Save the subsampled data
        with open(OUTPUT_FILE, "w") as f:
            json.dump(sampled_data, f, indent=4)

        # original vs subset answer types
        original_answer_types = Counter([sample["answer_type"] for sample in self.data])
        subset_answer_types = Counter([sample["answer_type"] for sample in sampled_data])
        print("Original answer types:", original_answer_types)
        print("Subset answer types:", subset_answer_types)

    def _subsample_number_data(self, question_ids, target_count):
        return [self.question_id_to_sample[qid] for qid in question_ids[:target_count]]

    def _subsample_yes_no_data(self, question_ids, target_count):
        # keep half "yes" and half "no" answers
        yes_question_ids = [qid for qid in question_ids if self.question_id_to_answer[qid] == "yes"]
        no_question_ids = [qid for qid in question_ids if self.question_id_to_answer[qid] == "no"]
        yes_samples = [self.question_id_to_sample[qid] for qid in random.sample(yes_question_ids, target_count // 2)]
        no_samples = [self.question_id_to_sample[qid] for qid in random.sample(no_question_ids, target_count // 2)]
        return yes_samples + no_samples

    def _subsample_unanswerable_data(self, question_ids, target_count):
        return random.sample([self.question_id_to_sample[qid] for qid in question_ids], target_count)

    def _subsample_other_data(self, question_ids, target_count):
        answers = [self.question_id_to_answer[qid] for qid in question_ids]
        answer_counts = Counter(answers)
        # Remove answers with less than 2 samples
        answer_counts = {ans: count for ans, count in answer_counts.items() if count >= 2}
        print(f"Found {len(answer_counts)} unique answers for other answer type")
        sampled_question_ids = []
        for answer in answer_counts:
            question_ids = [qid for qid, ans in self.question_id_to_answer.items() if ans == answer]
            sampled_question_ids.extend(random.sample(question_ids, 2))
        random.shuffle(sampled_question_ids)
        sampled_question_ids = sampled_question_ids[:target_count]
        return [self.question_id_to_sample[qid] for qid in sampled_question_ids]


if __name__ == "__main__":
    subsampler = VizWizSubsampler(INPUT_FILE, OUTPUT_FILE, target_answer_type_counts)
    subsampler.subsample()
