import argparse
import json
import os

import evaluate
import torch
from collate import Collator, SquadCollator
from datasets import load_dataset
from huggingface_hub import snapshot_download
from peft import PeftModel
from tqdm import tqdm
from transformers import MllamaForConditionalGeneration, MllamaProcessor

"""
python test.py \
	--hf_checkpoint_folder lorarank_32-loraalpha_64-lr_1e-4 \
	--hf_dataset_id gpantaz/flores200devtest \
	--hf_dataset_split devtest

python test.py \
	--hf_checkpoint_folder lorarank_32-loraalpha_64-lr_1e-4 \
	--hf_dataset_id gpantaz/ntrex128test \
	--hf_dataset_split test
"""


def load_model_and_processor(
    model_name: str, finetuning_path: str | None = None
) -> tuple[MllamaForConditionalGeneration | PeftModel, MllamaProcessor]:
    """Load model and processor with optional LoRA adapter"""
    print(f"Loading model: {model_name}")
    model = MllamaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = MllamaProcessor.from_pretrained(
        model_name,
        chat_template=True,
        padding=True,
        padding_side="left",
    )

    if finetuning_path and os.path.exists(finetuning_path):
        print(f"Loading LoRA adapter from '{finetuning_path}'...")
        model = PeftModel.from_pretrained(
            model,
            finetuning_path,
            is_adapter=True,
            torch_dtype=torch.bfloat16,
        )

    return model, processor  # type: ignore[report]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hf_repo_id",
        default="gpantaz/vae",
        help="Hugging Face repository ID containing model checkpoints",
    )

    parser.add_argument(
        "--hf_checkpoint_folder",
        default="lorarank_32-loraalpha_64-lr_1e-4",
        help="Hugging Face checkpoint folder within repo",
    )

    parser.add_argument(
        "--local_dir",
        default="hf_checkpoints",
        help="Local directory to download checkpoints",
    )

    parser.add_argument(
        "--hf_dataset_id",
        default="gpantaz/flores200devtest",
        choices=[
            "gpantaz/flores200devtest",
            "gpantaz/ntrex128test",
            "gpantaz/squadv2validation",
        ],
        help="Hugging Face dataset ID",
    )

    # TODO: remove the split dependency
    parser.add_argument(
        "--hf_dataset_split",
        default="devtest",
        choices=["devtest", "test", "validation"],
        help="Hugging Face dataset split",
    )

    parser.add_argument(
        "--task",
        default="wmt",
        choices=["wmt", "squad"],
        help="Task type",
    )

    parser.add_argument(
        "--src_image",
        default="br_image_aug",
        help="Source image column name",
    )

    parser.add_argument(
        "--context",
        default="br_context_image_aug",
        help="Context column name",
    )

    parser.add_argument(
        "--src_lang",
        default="en_question",
        help="Source language column name",
    )

    parser.add_argument(
        "--tgt_lang",
        default="en",
        choices=["en", "en_answer"],
        help="Target language column name",
    )

    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="Batch size for inference",
    )

    parser.add_argument(
        "--max_new_tokens",
        default=30,
        type=int,
        help="Maximum number of new tokens to generate",
    )

    parser.add_argument(
        "--num_workers",
        default=4,
        type=int,
        help="Number of workers for data loading",
    )

    parser.add_argument(
        "--last_only",
        action="store_true",
        help="Use only the last checkpoint",
    )

    parser.add_argument(
        "--download",
        action="store_true",
        help="Download checkpoints",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.download:
        pattern = f"{args.hf_checkpoint_folder}/*adapter*"
        _ = snapshot_download(
            repo_id=args.hf_repo_id,
            repo_type="model",
            local_dir=args.local_dir,
            allow_patterns=pattern,
            max_workers=args.num_workers,
            resume_download=True,
        )

        checkpoint_folder = os.path.join(args.local_dir, args.hf_checkpoint_folder)
    else:
        checkpoint_folder = args.hf_checkpoint_folder

    checkpoints = sorted(
        os.listdir(checkpoint_folder), key=lambda x: int(x.split("-")[-1])
    )
    if args.last_only:
        checkpoints = [checkpoints[-1]]
    else:
        print(f"Found {len(checkpoints)} checkpoints in {checkpoint_folder}")

    test_dataset = load_dataset(args.hf_dataset_id, split=args.hf_dataset_split)
    references = [[example[args.tgt_lang]] for example in test_dataset]
    for checkpoint in tqdm(checkpoints, total=len(checkpoints)):
        finetuning_path = os.path.join(checkpoint_folder, checkpoint)
        model, processor = load_model_and_processor(
            model_name="meta-llama/Llama-3.2-11B-Vision-Instruct",
            finetuning_path=finetuning_path,
        )

        if args.task == "wmt":
            collator = Collator(
                processor=processor,
                src_image=args.src_image,
                tgt_lang=args.tgt_lang,
                is_training=False,
            )
        else:
            collator = SquadCollator(
                processor=processor,
                context=args.context,
                src_lang=args.src_lang,
                tgt_lang=args.tgt_lang,
                is_training=False,
            )

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            collate_fn=collator,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        pbar = tqdm(
            test_dataloader,
            total=len(test_dataloader),
            desc=f"Predicting with {checkpoint}",
        )
        predictions = []
        for batch in pbar:
            inputs = {k: v.to(model.device) for k, v in batch.items()}
            batch_output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
            for output in batch_output:
                generated_response = output[inputs["input_ids"].shape[1] + 2 :]
                output_str = processor.decode(
                    generated_response, skip_special_tokens=True
                ).strip()

                predictions.append(output_str)

        chrf = evaluate.load("chrf")
        results = chrf.compute(
            predictions=predictions,
            references=references,
            word_order=2,
        )
        print(results)

        output_json = os.path.join(
            finetuning_path, f"{os.path.basename(args.hf_dataset_id)}_results.json"
        )
        with open(output_json, "w") as f:
            json.dump(
                {
                    "results": results,
                    "predictions": predictions,
                    "references": references,
                },
                f,
                indent=4,
            )

        # For some reason the gpu memory is not freed after each checkpoint
        del model
