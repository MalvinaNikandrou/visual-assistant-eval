import os
from dataclasses import dataclass, field
from typing import Literal

import torch
import transformers
from collate import Collator, SquadCollator
from datasets import Dataset, concatenate_datasets, load_dataset
from peft import LoraConfig, get_peft_model


def load_hf_datasets(dataset_ids: list[str], dataset_splits: list[str]) -> Dataset:
    datasets = []
    for dataset_id, dataset_split in zip(dataset_ids, dataset_splits):
        dataset = load_dataset(dataset_id)[dataset_split]  # type: ignore
        datasets.append(dataset)
    return concatenate_datasets(datasets)


def compute_trainable_params(model: torch.nn.Module) -> None:
    """Compute trainable parameters."""
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    train_params = sum([p.numel() for p in model_parameters])
    print(f"{sum([p.numel() for p in model.parameters()])} params and {train_params} trainable params")


@dataclass
class DataArguments:
    """Data arguments."""

    task: Literal["wmt", "squad"] = field(default="wmt")
    train_dataset: str = field(default="gpantaz/wmt2024train")
    train_split: str = field(default="train")
    eval_dataset: str = field(default="gpantaz/flores200dev")
    eval_split: str = field(default="dev")

    # Used by WMTCollator
    src_image: str = field(default="br_image_aug")
    # Used by SquadCollator
    context: str = field(default="br_context_image_aug")
    src_lang: str = field(default="en_question")
    # Used by both collators
    tgt_lang: str = field(default="en")


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

    train_dataset = load_dataset(data_args.train_dataset, split=data_args.train_split, num_proc=4)
    eval_dataset = load_dataset(data_args.eval_dataset, split=data_args.eval_split, num_proc=4)

    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    processor = transformers.AutoProcessor.from_pretrained(
        model_id,
        chat_template=True,
        padding=True,
    )
    processor.tokenizer.padding_side = "right"

    if data_args.task == "wmt":
        collator = Collator(
            processor=processor,
            src_image=data_args.src_image,
            tgt_lang=data_args.tgt_lang,
        )
    else:
        collator = SquadCollator(
            processor=processor,
            context=data_args.context,
            src_lang=data_args.src_lang,
            tgt_lang=data_args.tgt_lang,
        )

    # train_dataset = load_hf_datasets(["gpantaz/flores200devtest", "gpantaz/flores200devtest"])
    # eval_dataset = load_hf_datasets(["gpantaz/flores200devtest", "gpantaz/flores200devtest"])
    # train_dataset = load_dataset(data_args.train_dataset)[data_args.train_split]
    # eval_dataset = load_dataset(data_args.eval_dataset)[data_args.eval_split]
    # train_dataset = load_dataset("gpantaz/wmt2024train", split="train")
    # eval_dataset = load_dataset("gpantaz/flores200dev", split="dev")

    model = transformers.MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # quantization_config=config
    )

    lora_config = {
        "r": train_args.lora_rank,
        "lora_alpha": train_args.lora_alpha,
        "lora_dropout": 0.01,
        "target_modules": ["q_proj", "v_proj"],
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "inference_mode": False,
    }
    peft_config = LoraConfig(**lora_config)
    model = get_peft_model(model, peft_config)

    compute_trainable_params(model)

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
