import copy
from typing import Any
import torch
from dataclasses import dataclass, field
from datasets import load_dataset, concatenate_datasets, Dataset
from PIL import Image
import os
# from transformers import MllamaForConditionalGeneration, AutoProcessor
import transformers
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig
from dataclasses import asdict

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
    processor: transformers.AutoProcessor,
) -> dict[str, torch.Tensor]:
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
                labels[i] == processor.tokenizer.pad_token_id or labels[i] == 128256
            ):  #  128256 is image token index
                labels[i] = -100
        label_list.append(labels)
    batch["labels"] = torch.tensor(label_list)
    return batch



class Collator:
    def __init__(self, processor: transformers.AutoProcessor) -> None:
        self.processor = processor

    def __call__(self, examples: dict[str, Any]) -> dict[str, torch.Tensor]:
        images = [example["br_image_aug"] for example in examples]
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
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": example["en"],
                        }
                    ],
                },
            ]
            dialogs.append(dialog)
        return tokenize_dialogs(dialogs, images, self.processor)


def load_hf_datasets(dataset_ids: list[str], dataset_splits: list[str]) -> Dataset:
    datasets = []
    for dataset_id, dataset_split in zip(dataset_ids, dataset_splits):
        dataset = load_dataset(dataset_id)[dataset_split]
        datasets.append(dataset)
    return concatenate_datasets(datasets)

def compute_trainable_params(model: torch.nn.Module) -> None:
    """Compute trainable parameters."""
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    train_params = sum([p.numel() for p in model_parameters])
    print(
        f"{sum([p.numel() for p in model.parameters()])} params and {train_params} trainable params"
    )


@dataclass
class DataArguments:
    """Data arguments."""
        
    train_dataset: str = field(default="gpantaz/flores200dev")
    train_split: str = field(default="dev")
    eval_dataset: str = field(default="gpantaz/flores200dev")
    eval_split: str = field(default="dev")


@dataclass
class ModelArguments:
    """Model arguments."""
    
    model_id: str = field(default="meta-llama/Llama-3.2-11B-Vision-Instruct")
    lora_config_path: str = field(default="lora_config.json")


@dataclass
class TrainArgs(transformers.TrainingArguments):
    """Training arguments."""

    output_dir: str = field(default="checkpoints")
    per_device_train_batch_size: int = field(default=4)
    per_device_eval_batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(default=16)
    logging_steps: int = field(default=1)
    save_strategy: str = field(default="steps")
    save_steps: float = field(default=0.05)
    num_train_epochs: int = field(default=1)
    learning_rate: float = field(default=1e-4)
    weight_decay: float = field(default=0.1)
    warmup_ratio: float = field(default=0.1)
    lr_scheduler_type: str = field(default="linear")
    bf16: bool = field(default=True)
    fp16: bool = field(default=False)
    gradient_checkpointing: bool = field(default=False)
    # deepspeed: str = field(default="configs/trainer/zero2.json")
    save_total_limit: int = field(default=30)
    load_best_model_at_end: bool = field(default=True)
    log_level: str = field(default="debug")
    save_safetensors: bool = field(default=True)
    evaluation_strategy: str = field(default="steps")
    eval_steps: float = field(default=0.05)
    seed: int = field(default=12345)
    data_seed: int = field(default=12345)
    dataloader_num_workers: int = field(default=4)
    logging_nan_inf_filter: bool = field(default=False)
    run_name: str = field(default="llama-3.2-11b-vision-instruct")
    project_name: str = field(default="visual-assistant-eval")
    remove_unused_columns: bool = field(default=False)
    lora_rank: int = field(default=8)
    lora_alpha: int = field(default=16)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainArgs))
    model_args, data_args, train_args = parser.parse_args_into_dataclasses()
    
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    from transformers import BitsAndBytesConfig

    # config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    # )

    model = transformers.MllamaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto", #quantization_config=config
    )

    # import json
    # with open(model_args.lora_config_path, "r") as f:
    #     lora_config = json.load(f)
    lora_config = {
        "r": train_args.lora_rank,
        "lora_alpha": train_args.lora_alpha,
        "lora_dropout": 0.01,
        "target_modules": [
            "q_proj",
            "v_proj"
        ],
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "inference_mode": False
    }
    peft_config = LoraConfig(**lora_config)
    model = get_peft_model(model, peft_config)

    compute_trainable_params(model)
    # import subprocess
    # f = open("nvidia-smi.txt", "w")
    # subprocess.call("nvidia-smi".split(), stdout=f)
    # breakpoint()

    processor = transformers.AutoProcessor.from_pretrained(
        model_id, chat_template=True, padding=True,
    )
    processor.tokenizer.padding_side = "right"

    collator = Collator(processor)

    # train_dataset = load_hf_datasets(["gpantaz/flores200devtest", "gpantaz/flores200devtest"])
    # eval_dataset = load_hf_datasets(["gpantaz/flores200devtest", "gpantaz/flores200devtest"])
    # train_dataset = load_dataset(data_args.train_dataset)[data_args.train_split]
    # eval_dataset = load_dataset(data_args.eval_dataset)[data_args.eval_split]
    train_dataset = load_dataset("gpantaz/wmt2024train", split="train")
    eval_dataset = load_dataset("gpantaz/flores200dev", split="dev")

    # train_dataloader = torch.utils.data.DataLoader(
    #     train_dataset, collate_fn=collator, batch_size=
    # )

    # eval_dataloader = torch.utils.data.DataLoader(
    #     eval_dataset, collate_fn=collator, batch_size=2
    # )
    
    if train_args.project_name is not None:
        os.environ["WANDB_PROJECT"] = train_args.project_name

    trainer = transformers.Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    trainer.train()

if __name__ == "__main__":
    train()