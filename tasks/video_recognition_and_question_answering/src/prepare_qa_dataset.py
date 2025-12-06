import json
import os
from random import shuffle
from typing import Any

import torch
import transformers

EXTRACT_COMMON_WORD_PROMPT = """You will be given you a list of objects, and you have to answer with one short word or phrase that can be used to describe the group.

Examples
Objects: [watch, wrist watch, apple watch, apple wath, risk watch, my apple watch]
Answer: watch

Objects: [black small wallet, my purse, my wallet, ladies purse, money pouch, coin purse, wallet for bus pass cards and money, i d wallet, ipod in wallet, walletv, wallet, purse]
Answer: wallet

Objects: [orbit braille reader and notetaker, orbit reader 20 braille display, braillepen slim braille keyboard, braille orbit reader, braille note, my braille displat]
Answer: braille reading device


Generate the answer for the following:
Objects: {group_of_objects}
"""


GENERATE_QUESTION_PROMPT = """You will be given a list of objects and a common label that describes the group. Your task is to generate a question that can be asked to identify an instance of this group in a video.

Examples:
Objects: [slippers, nike trainers, my shoes, boot, trainers, trainer shoe, slipper, my trainers, shoes, running shoes]
Group: shoes
Question: What type of clothing do you see in the video?

Objects: [orbit braille reader and notetaker, orbit reader 20 braille display, braillepen slim braille keyboard, braille orbit reader, braille note, my braille displat]
Group: braille reading device
Question: What kind of device was there?

Objects: [black small wallet, my purse, my wallet, ladies purse, money pouch, coin purse, wallet for bus pass cards and money, i d wallet, ipod in wallet, walletv, wallet, purse]
Group: wallet
Question: What type of accessory appears in the video?

Generate the question for the following:
Objects: {group_of_objects}
Group: {group_label}
"""


class BaseProcessor:
    def __init__(
        self,
        model: str,
        data_file: str,
        torch_dtype: torch.dtype,
        max_new_tokens: int = 64,
    ) -> None:
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            model_kwargs={"torch_dtype": torch_dtype},
            device_map="auto",
        )
        self.max_new_tokens = max_new_tokens
        self.data = self._load_data(data_file)

    def _load_data(self):
        raise NotImplementedError

    def get_model_output(self, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful, careful and concise assistant.",
            },
            {"role": "user", "content": prompt},
        ]

        outputs = self.pipeline(
            messages,
            max_new_tokens=self.max_new_tokens,
        )
        prompt = outputs[0]["generated_text"][-1]["content"]
        return prompt


class ClusterLabelProcessor(BaseProcessor):
    """
    Generate a common label for a cluster of objects.
    """

    def __init__(self, model: str, data_file: str, torch_dtype: torch.dtype) -> None:
        super().__init__(model, data_file, torch_dtype)
        self.prompt = EXTRACT_COMMON_WORD_PROMPT
        self.output_file = os.path.join(os.path.dirname(data_file), "orbit_clusters_with_labels.txt")

    def _load_data(self, data_file: str) -> list[str]:
        # Each row is a cluster of objects written as a list
        data = []
        with open(data_file, "r") as f:
            data = f.readlines()
        data = [line.strip() for line in data]
        return data

    def run(self) -> None:
        gourp_label = []
        for group_of_objects in self.data:
            sample_prompt = self.prompt.format(group_of_objects=group_of_objects)
            output = self.get_model_output(sample_prompt)
            gourp_label.append(f"{output.strip()}\t{group_of_objects}")

        with open(self.output_file, "w") as f:
            f.write("\n".join(gourp_label))


class VideoQAGenerator(BaseProcessor):
    """
    Generate a `What type` question for a cluster of objects.
    """

    def __init__(self, model: str, data_file: str, torch_dtype: torch.dtype) -> None:
        super().__init__(model, data_file, torch_dtype, max_new_tokens=128)

        self.prompt = GENERATE_QUESTION_PROMPT
        self.output_file = os.path.join(os.path.dirname(data_file), "orbit_recognition_clusters_with_questions.json")

    def _load_data(self, data_file: str) -> list[str]:
        # Each row is a cluster of objects written as a list
        data = []
        with open(data_file, "r") as f:
            for line in f:
                line = line.strip()
                parts = line.split("\t")
                assert len(parts) == 2  # Each line is expected to have a group label and a list of objects
                data.append(line)
        return data

    def run(self) -> None:
        data = []
        for sample in self.data:
            group_label, group_of_objects = sample.split("\t")
            sample_prompt = self.prompt.format(group_of_objects=group_of_objects, group_label=group_label)
            output = self.get_model_output(sample_prompt).strip()
            for word in group_of_objects.split(","):
                sample = {
                    "object": word.strip(),
                    "group": group_label,
                    "question": output,
                }
                data.append(sample)
        with open(self.output_file, "w") as f:
            json.dump(data, f, indent=4)


if __name__ == "__main__":
    word_processor = ClusterLabelProcessor(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        data_file="data/object_clusters_full.txt",
        torch_dtype=torch.bfloat16,
    )
    word_processor.run()

    supercategory_processor = VideoQAGenerator(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        data_file="data/orbit_clusters_with_labels.txt",
        torch_dtype=torch.bfloat16,
    )
    supercategory_processor.run()
