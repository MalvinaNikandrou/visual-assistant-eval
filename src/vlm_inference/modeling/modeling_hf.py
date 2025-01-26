import logging
from typing import Any, Callable, Dict, Optional, Tuple, Type
from qwen_vl_utils import process_vision_info

import torch
from outlines.integrations.transformers import JSONPrefixAllowedTokens
from PIL import Image
from pydantic import BaseModel as PydanticBaseModel
from transformers.feature_extraction_utils import BatchFeature
from transformers import GenerationConfig

from ..dataset.dataset_base import ImageExample, VideoExample
from ..utils.misc import torch_dtype_from_str
from ..utils.usage_tracking import UsageMetadata
from .modeling_base import VisionLanguageModel

logger = logging.getLogger(__name__)


class HfModel(VisionLanguageModel):

    def __init__(
        self,
        name: str,
        generation_kwargs: Dict[str, Any],
        json_mode: bool,
        dtype: str,
        model_cls: Callable,
        processor_cls: Callable,
        strip_prompt: bool = False,
        postprocess_fn: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(name, generation_kwargs, json_mode)

        self.model = model_cls(
            pretrained_model_name_or_path=self.name,
            torch_dtype=torch_dtype_from_str(dtype),
        )

        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Model loaded on {self.model.device}")
        self.processor = processor_cls(pretrained_model_name_or_path=self.name)
        self.strip_prompt = strip_prompt
        self.postprocess_fn = postprocess_fn

    def _extract_features(self, example: ImageExample) -> BatchFeature:
        inputs = self.processor(
            images=Image.open(example.image_path).convert("RGB"),
            text=example.prompt,
            return_tensors="pt",
            padding=True,
        )  # .to(self.model.device, dtype=self.model.dtype
        # inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        for key, values in inputs.items():
            if values.dtype == torch.float32:
                inputs[key] = values.to(self.model.dtype)
        return inputs

    def generate(
        self, example: ImageExample, json_schema: Optional[Type[PydanticBaseModel]] = None
    ) -> Tuple[str, UsageMetadata]:
        features = self._extract_features(example)

        prefix_allowed_tokens_fn = (
            JSONPrefixAllowedTokens(
                schema=json_schema,
                tokenizer_or_pipe=self.processor,
                whitespace_pattern=r" ?",
            )
            if json_schema is not None
            else None
        )
        generated_tokens = self.model.generate(
            **features,
            tokenizer=self.processor.tokenizer,
            **self.generation_kwargs,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )

        if self.strip_prompt:
            generated_tokens = generated_tokens[:, features["input_ids"].shape[1] :]

        generated_tokens = generated_tokens.cpu()

        generated_text = self.processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()
        if self.postprocess_fn is not None:
            generated_text = self.postprocess_fn(generated_text)
        usage_metadata = UsageMetadata(
            input_token_count=features["input_ids"].shape[1],
            output_token_count=generated_tokens.shape[1],
        )

        return generated_text, usage_metadata


class MolmoModel(HfModel):

    def _extract_features(self, example: ImageExample) -> BatchFeature:
        inputs = self.processor(
            images=Image.open(example.image_path).convert("RGB"),
            text=example.prompt,
            return_tensors="pt",
            padding=True,
        )  # .to(self.model.device, dtype=self.model.dtype
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}
        for key, values in inputs.items():
            if values.dtype == torch.float32:
                inputs[key] = values.to(self.model.dtype)
        return inputs

    def generate(
        self, example: ImageExample, json_schema: Optional[Type[PydanticBaseModel]] = None
    ) -> Tuple[str, UsageMetadata]:
        features = self._extract_features(example)

        generated_tokens = self.model.generate_from_batch(
            features,
            GenerationConfig(
                max_new_tokens=self.generation_kwargs.get("max_new_tokens"),
                stop_strings="<|endoftext|>",
                do_sample=self.generation_kwargs.get("do_sample"),
                temperature=self.generation_kwargs.get("temperature"),
                top_p=self.generation_kwargs.get("top_p"),
            ),
            tokenizer=self.processor.processor.tokenizer,
        )

        if self.strip_prompt:
            generated_tokens = generated_tokens[:, features["input_ids"].shape[1] :]

        generated_tokens = generated_tokens.cpu()

        generated_text = self.processor.processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[
            0
        ].strip()
        if self.postprocess_fn is not None:
            generated_text = self.postprocess_fn(generated_text)
        usage_metadata = UsageMetadata(
            input_token_count=features["input_ids"].shape[1],
            output_token_count=generated_tokens.shape[1],
        )

        return generated_text, usage_metadata


class VideoHfModel(HfModel):

    def _prep_video(self, video_path: str) -> Dict[str, Any]:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 360 * 420,
                        "fps": 4.0,
                    },
                ],
            }
        ]
        _, videos = process_vision_info(messages)
        return videos

    def _extract_features(self, example: VideoExample) -> BatchFeature:
        videos = self._prep_video(example.image_path)
        inputs = self.processor(videos=videos, text=example.prompt, return_tensors="pt", padding=True)
        # .to(self.model.device, dtype=self.model.dtype
        # inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        for key, values in inputs.items():
            if values.dtype == torch.float32:
                inputs[key] = values.to(self.model.dtype)
        return inputs
