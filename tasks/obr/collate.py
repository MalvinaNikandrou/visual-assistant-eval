import copy
from typing import Any

import torch
from PIL import Image
from transformers import MllamaProcessor
from transformers.feature_extraction_utils import BatchFeature

# Adapted from https://github.com/meta-llama/llama-cookbook/blob/main/getting-started/finetuning/datasets/ocrvqa_dataset.py


# check system prompt token seq or user prompt token seq is in the current token list
def check_header(targets: torch.Tensor, seq: torch.Tensor) -> bool:
    for i in range(len(seq) - 3):
        if seq[i : i + 3] in targets:
            return True
    return False


def replace_target(target: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
    for i in range(len(seq) - 3):
        if seq[i : i + 3] == target:
            seq[i], seq[i + 1], seq[i + 2] = -100, -100, -100
    return seq


def tokenize_dialogs(
    dialogs: list[dict[str, Any]],
    images: list[Image.Image],
    processor: MllamaProcessor,
) -> BatchFeature:
    text_prompt = processor.apply_chat_template(dialogs)
    text_prompt = [prompt.replace("<|begin_of_text|>", "") for prompt in text_prompt]
    batch = processor(
        images=images, text=text_prompt, padding=True, return_tensors="pt"
    )
    label_list = []
    for i in range(len(batch["input_ids"])):
        dialog_tokens = batch["input_ids"][i].tolist()
        labels = copy.copy(dialog_tokens)
        eot_indices = [i for i, n in enumerate(labels) if n == 128009]
        last_idx = 0
        # system prompt header "<|start_header_id|>system<|end_header_id|>" has been tokenized to [128006, 9125, 128007]
        # user prompt header "<|start_header_id|>user<|end_header_id|>" has been tokenized to [128006, 882, 128007]
        prompt_header_seqs = [[128006, 9125, 128007], [128006, 882, 128007]]
        for n, idx in enumerate(eot_indices):
            current_seq = labels[last_idx : idx + 1]
            if check_header(prompt_header_seqs, current_seq):
                # found prompt header, indicating that this seq should be masked
                labels[last_idx : idx + 1] = [-100] * (idx - last_idx + 1)
            else:
                last_idx = idx + 1
            #  Mask all the assistant header prompt <|start_header_id|>assistant<|end_header_id|>, which has been tokenized to [128006, 78191, 128007]
        assistant_header_seq = [128006, 78191, 128007]
        labels = replace_target(assistant_header_seq, labels)
        # Mask the padding token and image token 128256
        for i in range(len(labels)):
            if (
                labels[i] == processor.tokenizer.pad_token_id or labels[i] == 128256  # type: ignore
            ):  #  128256 is image token index
                labels[i] = -100
        label_list.append(labels)
    batch["labels"] = torch.tensor(label_list)
    return batch


class Collator:
    def __init__(
        self,
        processor: MllamaProcessor,
        src_image: str = "br_image_aug",
        tgt_lang: str = "en",
        is_training: bool = True,
    ) -> None:
        self.processor = processor
        self.src_image = src_image
        self.tgt_lang = tgt_lang
        self.is_training = is_training

    def __call__(self, examples: dict[str, Any]) -> BatchFeature:
        images = [example[self.src_image] for example in examples]
        dialogs = []
        for example in examples:
            dialog = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Translate the Braille to English."},
                    ],
                },
            ]
            if self.is_training:
                dialog.append(
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": example[self.tgt_lang],
                            }
                        ],
                    },
                )
            dialogs.append(dialog)
        return tokenize_dialogs(dialogs, images, self.processor)


class SquadCollator:
    def __init__(
        self,
        processor: MllamaProcessor,
        context: str = "br_context_image_aug",
        src_lang: str = "en_question",
        tgt_lang: str = "en_answer",
        is_training: bool = True,
    ) -> None:
        self.processor = processor
        self.context = context
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.prompt = """Answer the following question based on the image.\nIf the question is not answerable, output 'unanswerable'.\n{question}"""
        self.is_training = is_training

    def __call__(self, examples: dict[str, Any]) -> BatchFeature:
        images = [example[self.context] for example in examples]
        dialogs = []
        for example in examples:
            dialog = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": self.prompt.format(question=example[self.src_lang]),
                        },
                    ],
                },
            ]
            if self.is_training:
                dialog.append(
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": example[self.tgt_lang],
                            }
                        ],
                    },
                )

            dialogs.append(dialog)
        return tokenize_dialogs(dialogs, images, self.processor)
