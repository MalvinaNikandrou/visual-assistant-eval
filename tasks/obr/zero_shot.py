import argparse
import json
from dataclasses import dataclass

import evaluate
import torch
from datasets import Dataset, load_dataset
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoProcessor,
    GenerationConfig,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    MllamaForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
)

PLAIN_PROMPT = """
Translate the Grade 1 Braille shown in the image to plain English.
The result should be in a JSON format, {"text": "translated text"}.
It is important to be consistent with the format, do not include any additional information.
Only output the requested JSON and nothing else.
"""

ALPHABET_PROMPT = """
The Grade-1 braille alphabet, braille numbers, braille punctuation and special symbols characters are constructed from six dots.
These braille dots are positioned like the figure six on a die, in a grid of two parallel vertical lines of three dots each.
From the six dots that make up the basic grid, 64 different configurations can be created.
The english characters, numbers, and punctuation and parantheses are represented by the following braille
characters (in unicode):

'a': '⠁', 'b': '⠃', 'c': '⠉', 'd': '⠙', 'e': '⠑', 'f': '⠋', 'g': '⠛', 'h': '⠓', 'i': '⠊', 'j':
'⠚', 'k': '⠅', 'l': '⠇', 'm': '⠍', 'n': '⠝', 'o': '⠕', 'p': '⠏', 'q': '⠟', 'r': '⠗', 's': '⠎',
't': '⠞', 'u': '⠥', 'v': '⠧', 'w': '⠺', 'x': '⠭', 'y': '⠽', 'z': '⠵', '0': '⠚', '1': '⠁', '2':
'⠃', '3': '⠉', '4': '⠙', '5': '⠑', '6': '⠋', '7': '⠛', '8': '⠓', '9': '⠊', ' ': ' ', '!': '⠖',
'$': '⠲', "'": '⠄', '(': '⠶', ')': '⠶', ',': '⠂', '-': '⠤', '.': '⠲', '/': '⠌', ':': '⠒', ';':
'⠰', '?': '⠦'

Additionally the '⠠' symbol denotes capitalization, while '⠼' denotes the beginning of a number.

Translate the Grade 1 Braille shown in the image to plain English.
The result should be in a JSON format, {"text": "translated text"}.
It is important to be consistent with the format, do not include any additional information.
Only output the requested JSON and nothing else.
"""

# For example the sentence 'Hello, World! 42' would be represented in braille as:
# '⠠⠓⠑⠇⠇⠕⠂ ⠠⠺⠕⠗⠇⠙⠖ ⠼⠙⠼⠃'

PROMPTS = [
    PLAIN_PROMPT,
    ALPHABET_PROMPT,
]


@dataclass
class GenerationParameters:
    do_sample: bool = False
    max_new_tokens: int = 60


def evaluate_idefics3_model(
    dataset: Dataset,
    hf_model_id: str,
    generation_params: GenerationParameters,
    prompt: str,
    src_image: str = "br_image_aug",
    tgt_lang: str = "en",
) -> tuple[list[str], list[str], float]:
    model = AutoModelForVision2Seq.from_pretrained(
        hf_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(hf_model_id)

    predictions = []
    references = []
    count = 0
    pbar = tqdm(dataset, total=len(dataset), desc=f"Predicting with {hf_model_id}")
    for example in pbar:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(
            text=input_text, images=[example[src_image]], return_tensors="pt"
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        output = model.generate(
            **inputs,
            max_new_tokens=generation_params.max_new_tokens,
            do_sample=generation_params.do_sample,
        )

        generated_response = processor.tokenizer.decode(
            output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()

        if '{"text"' in generated_response:
            try:
                left_index = generated_response.index('{"text"')
                if '"}' in generated_response[left_index:]:
                    right_index = generated_response.index('"}', left_index)
                    json_str = generated_response[left_index : right_index + 2]
                else:
                    json_str = generated_response[left_index:] + '"}'
                output_str = json.loads(json_str)["text"]
            except:
                count += 1
                output_str = ""
        else:
            count += 1
            output_str = ""

        predictions.append(output_str)
        references.append([example["en"]])
        pbar.set_postfix({"missed": count})

    return predictions, references, round(count / len(dataset) * 100, 2)


def evaluate_llavav1_6_model(
    dataset: Dataset,
    hf_model_id: str,
    generation_params: GenerationParameters,
    prompt: str,
    src_image: str = "br_image_aug",
    tgt_lang: str = "en",
) -> tuple[list[str], list[str], float]:
    model = LlavaNextForConditionalGeneration.from_pretrained(
        hf_model_id, torch_dtype="auto", device_map="auto"
    )

    processor = LlavaNextProcessor.from_pretrained(hf_model_id)

    predictions = []
    references = []
    count = 0
    pbar = tqdm(dataset, total=len(dataset), desc=f"Predicting with {hf_model_id}")
    for example in pbar:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},
                ],
            },
        ]
        input_prompt = processor.apply_chat_template(
            messages, add_generation_prompt=True
        )

        inputs = processor(
            images=example[src_image], text=input_prompt, return_tensors="pt"
        ).to(model.device)

        # autoregressively complete prompt
        output = model.generate(
            **inputs,
            do_sample=generation_params.do_sample,
            max_new_tokens=generation_params.max_new_tokens,
        )

        generated_response = processor.tokenizer.decode(
            output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()

        try:
            left_index = generated_response.index("{")
            right_index = generated_response.index("}") + 1
            json_str = generated_response[left_index:right_index]
            output_str = json.loads(json_str)["text"]
        except:
            count += 1
            output_str = ""

        # print(generated_response)
        # print(output_str)
        # print(example["en"])
        # print("-----")
        # breakpoint()

        predictions.append(output_str)
        references.append([example["en"]])
        pbar.set_postfix({"missed": count})

    return predictions, references, round(count / len(dataset) * 100, 2)


def evaluate_llama_model(
    dataset: Dataset,
    hf_model_id: str,
    generation_params: GenerationParameters,
    prompt: str,
    src_image: str = "br_image_aug",
    tgt_lang: str = "en",
) -> tuple[list[str], list[str], float]:
    model = MllamaForConditionalGeneration.from_pretrained(
        hf_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(hf_model_id)

    predictions = []
    references = []
    count = 0
    pbar = tqdm(dataset, total=len(dataset), desc=f"Predicting with {hf_model_id}")
    for example in pbar:
        references.append([example["en"]])

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(
            example[src_image],
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(model.device)

        output = model.generate(
            **inputs,
            max_new_tokens=generation_params.max_new_tokens,
            do_sample=generation_params.do_sample,
        )

        generated_response = processor.tokenizer.decode(
            output[0, inputs.input_ids.shape[1] :]
        )

        # output_str = postprocess_output(generated_response)
        if '{"text"' in generated_response:
            try:
                left_index = generated_response.index('{"text"')
                if '"}' in generated_response[left_index:]:
                    right_index = generated_response.index('"}', left_index)
                    json_str = generated_response[left_index : right_index + 2]
                else:
                    json_str = generated_response[left_index:] + '"}'
                output_str = json.loads(json_str)["text"]
            except:
                count += 1
                output_str = ""
        else:
            count += 1
            output_str = ""

        predictions.append(output_str)
        pbar.set_postfix({"missed": count})

    return predictions, references, round(count / len(dataset) * 100, 2)


def evaluate_molmo_model(
    dataset: Dataset,
    hf_model_id: str,
    generation_params: GenerationParameters,
    prompt: str,
    src_image: str = "br_image_aug",
    tgt_lang: str = "en",
) -> tuple[list[str], list[str], float]:
    processor = AutoProcessor.from_pretrained(
        hf_model_id, trust_remote_code=True, torch_dtype="auto", device_map="auto"
    )

    model = AutoModelForCausalLM.from_pretrained(
        hf_model_id, trust_remote_code=True, torch_dtype="auto", device_map="auto"
    )

    predictions = []
    references = []
    count = 0
    pbar = tqdm(dataset, total=len(dataset), desc=f"Predicting with {hf_model_id}")
    for example in pbar:
        inputs = processor.process(images=[example[src_image]], text=prompt)

        # move inputs to the correct device and make a batch of size 1
        inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

        # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(
                do_sample=generation_params.do_sample,
                max_new_tokens=generation_params.max_new_tokens,
                stop_strings="<|endoftext|>",
            ),
            tokenizer=processor.tokenizer,
        )

        generated_response = processor.tokenizer.decode(
            output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        # output_str = postprocess_output(generated_response)
        # if '{"text"' in generated_response:
        #     try:
        #         left_index = generated_response.index('{"text"')
        #         if '"}' in generated_response[left_index:]:
        #             right_index = generated_response.index('"}', left_index)
        #             json_str = generated_response[left_index : right_index + 2]
        #         else:
        #             json_str = generated_response[left_index:] + '"}'
        #         output_str = json.loads(json_str)["text"]
        #     except:
        #         count += 1
        #         output_str = ""
        # else:
        #     count += 1
        #     output_str = ""
        try:
            left_index = generated_response.index("{")
            right_index = generated_response.index("}") + 1
            json_str = generated_response[left_index:right_index]
            output_str = json.loads(json_str)["text"]
        except:
            count += 1
            output_str = ""

        # print(generated_response)
        # print(output_str)
        # print(example["en"])
        # print("-----")
        # breakpoint()

        predictions.append(output_str)
        references.append([example["en"]])
        pbar.set_postfix({"missed": count})

    return predictions, references, round(count / len(dataset) * 100, 2)


def evaluate_phi3vision_model(
    dataset: Dataset,
    hf_model_id: str,
    generation_params: GenerationParameters,
    prompt: str,
    src_image: str = "br_image_aug",
    tgt_lang: str = "en",
) -> tuple[list[str], list[str], float]:
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_id,
        device_map="cuda",
        trust_remote_code=True,
        torch_dtype="auto",
        _attn_implementation="eager",
    )  # use _attn_implementation='eager' to disable flash attention

    processor = AutoProcessor.from_pretrained(hf_model_id, trust_remote_code=True)
    predictions = []
    references = []
    count = 0
    pbar = tqdm(dataset, total=len(dataset), desc=f"Predicting with {hf_model_id}")
    for example in pbar:
        messages = [
            {"role": "user", "content": f"<|image_1|>\n{prompt}"},
        ]
        input_text = processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = processor(input_text, [example[src_image]], return_tensors="pt").to(
            model.device
        )

        generate_ids = model.generate(
            **inputs,
            eos_token_id=processor.tokenizer.eos_token_id,
            do_sample=generation_params.do_sample,
            max_new_tokens=generation_params.max_new_tokens,
        )
        generated_response = processor.tokenizer.batch_decode(
            generate_ids[:, inputs["input_ids"].shape[1] :]
        )[0].strip()

        try:
            left_index = generated_response.index("{")
            right_index = generated_response.index("}") + 1
            json_str = generated_response[left_index:right_index]
            output_str = json.loads(json_str)["text"]
        except:
            count += 1
            output_str = ""

        # print(generated_response)
        # print(output_str)
        # print(example["en"])
        # print("-----")
        # breakpoint()

        predictions.append(output_str)
        references.append([example["en"]])
        pbar.set_postfix({"missed": count})

    return predictions, references, round(count / len(dataset) * 100, 2)


def evaluate_qwen2vl_model(
    dataset: Dataset,
    hf_model_id: str,
    generation_params: GenerationParameters,
    prompt: str,
    src_image: str = "br_image_aug",
    tgt_lang: str = "en",
) -> tuple[list[str], list[str], float]:
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        hf_model_id, torch_dtype="auto", device_map="auto"
    )

    processor = AutoProcessor.from_pretrained(hf_model_id)

    predictions = []
    references = []
    count = 0
    pbar = tqdm(dataset, total=len(dataset), desc=f"Predicting with {hf_model_id}")
    for example in pbar:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": example[src_image],
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        generated_ids = model.generate(
            **inputs,
            do_sample=generation_params.do_sample,
            max_new_tokens=generation_params.max_new_tokens,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        generated_response = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        if '{"text"' in generated_response:
            try:
                left_index = generated_response.index('{"text"')
                if '"}' in generated_response[left_index:]:
                    right_index = generated_response.index('"}', left_index)
                    json_str = generated_response[left_index : right_index + 2]
                else:
                    json_str = generated_response[left_index:] + '"}'
                output_str = json.loads(json_str)["text"]
            except:
                count += 1
                output_str = ""
        else:
            count += 1
            output_str = ""

        print(output_str)
        print(example["en"])
        print("-----")

        predictions.append(output_str)
        references.append([example["en"]])
        pbar.set_postfix({"missed": count})

    return predictions, references, round(count / len(dataset) * 100, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hf_model_id",
        default="meta-llama/Llama-3.2-11B-Vision-Instruct",
        help="Hugging Face model ID",
        choices=[
            # "google/paligemma-3b-mix-448",
            "HuggingFaceM4/Idefics3-8B-Llama3",
            "llava-hf/llava-v1.6-mistral-7b-hf",
            "meta-llama/Llama-3.2-11B-Vision-Instruct",
            "allenai/Molmo-7B-D-0924",
            "microsoft/Phi-3-vision-128k-instruct",
            "Qwen/Qwen2-VL-7B-Instruct",
        ],
    )

    parser.add_argument(
        "--hf_dataset_id",
        default="gpantaz/flores200devtest",
        choices=["gpantaz/flores200devtest", "gpantaz/ntrex128test"],
        help="Hugging Face dataset ID",
    )

    # TODO: remove the split dependency
    parser.add_argument(
        "--hf_dataset_split",
        default="devtest",
        choices=["devtest", "test"],
        help="Hugging Face dataset split",
    )

    parser.add_argument(
        "--prompt_index",
        default=0,
        type=int,
        choices=list(range(len(PROMPTS))),
        help="Index of the prompt to use",
    )

    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Use sampling for generation",
    )

    parser.add_argument(
        "--max_new_tokens",
        default=100,
        type=int,
        help="Maximum number of new tokens to generate",
    )

    parser.add_argument(
        "--output_file",
        default="output.json",
        help="Output file for predictions",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    test_dataset = load_dataset(args.hf_dataset_id, split=args.hf_dataset_split)
    chrf = evaluate.load("chrf")

    if args.hf_model_id == "HuggingFaceM4/Idefics3-8B-Llama3":
        predictions, references, failed_parsed_perc = evaluate_idefics3_model(
            test_dataset,
            args.hf_model_id,
            GenerationParameters(
                do_sample=args.do_sample, max_new_tokens=args.max_new_tokens
            ),
            PROMPTS[args.prompt_index],
        )

    elif args.hf_model_id == "llava-hf/llava-v1.6-mistral-7b-hf":
        predictions, references, failed_parsed_perc = evaluate_llavav1_6_model(
            test_dataset,
            args.hf_model_id,
            GenerationParameters(
                do_sample=args.do_sample, max_new_tokens=args.max_new_tokens
            ),
            PROMPTS[args.prompt_index],
        )

    elif args.hf_model_id == "meta-llama/Llama-3.2-11B-Vision-Instruct":
        predictions, references, failed_parsed_perc = evaluate_llama_model(
            test_dataset,
            args.hf_model_id,
            GenerationParameters(
                do_sample=args.do_sample, max_new_tokens=args.max_new_tokens
            ),
            PROMPTS[args.prompt_index],
        )

    elif args.hf_model_id == "allenai/Molmo-7B-D-0924":
        predictions, references, failed_parsed_perc = evaluate_molmo_model(
            test_dataset,
            args.hf_model_id,
            GenerationParameters(
                do_sample=args.do_sample, max_new_tokens=args.max_new_tokens
            ),
            PROMPTS[args.prompt_index],
        )

    elif args.hf_model_id == "microsoft/Phi-3-vision-128k-instruct":
        predictions, references, failed_parsed_perc = evaluate_phi3vision_model(
            test_dataset,
            args.hf_model_id,
            GenerationParameters(
                do_sample=args.do_sample, max_new_tokens=args.max_new_tokens
            ),
            PROMPTS[args.prompt_index],
        )

    elif args.hf_model_id == "Qwen/Qwen2-VL-7B-Instruct":
        predictions, references, failed_parsed_perc = evaluate_qwen2vl_model(
            test_dataset,
            args.hf_model_id,
            GenerationParameters(
                do_sample=args.do_sample, max_new_tokens=args.max_new_tokens
            ),
            PROMPTS[args.prompt_index],
        )

    else:
        raise NotImplementedError(f"Model {args.hf_model_id} not implemented")

    results = chrf.compute(
        predictions=predictions,
        references=references,
        word_order=2,
    )

    with open(args.output_file, "w") as fp:
        json.dump(
            {
                "results": results,
                "predictions": predictions,
                "references": references,
                "failed_parsed_perc": failed_parsed_perc,
            },
            fp,
            indent=4,
        )
