import logging
from typing import Any, Callable, Dict, Optional, Tuple, Type

import av
import numpy as np
import torch
from decord import VideoReader, cpu
from outlines.integrations.transformers import JSONPrefixAllowedTokens
from PIL import Image
from pydantic import BaseModel as PydanticBaseModel
from qwen_vl_utils import process_vision_info
from transformers import GenerationConfig
from transformers.feature_extraction_utils import BatchFeature

from ..dataset.dataset_base import ImageExample, VideoExample
from ..utils.misc import torch_dtype_from_str
from ..utils.usage_tracking import UsageMetadata
from .modeling_base import VisionLanguageModel

logger = logging.getLogger(__name__)
MAX_NUM_FRAMES = 64  # if cuda OOM set a smaller number


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
        self,
        example: ImageExample,
        json_schema: Optional[Type[PydanticBaseModel]] = None,
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
        self,
        example: ImageExample,
        json_schema: Optional[Type[PydanticBaseModel]] = None,
    ) -> Tuple[str, UsageMetadata]:
        features = self._extract_features(example)

        generated_tokens = self.model.generate_from_batch(
            features,
            GenerationConfig(
                max_new_tokens=self.generation_kwargs.get("max_new_tokens"),
                stop_strings="<|endoftext|>",
                do_sample=self.generation_kwargs.get("do_sample", "False"),
                temperature=self.generation_kwargs.get("temperature"),
                top_p=self.generation_kwargs.get("top_p", "1.0"),
                top_k=self.generation_kwargs.get("top_k", "0"),
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


class QwenVideoHfModel(HfModel):
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


class LlaVANextVideoHfModel(HfModel):
    def read_video_pyav(self, container, indices):
        """
        Decode the video with PyAV decoder.
        Args:
            container (`av.container.input.InputContainer`): PyAV container.
            indices (`List[int]`): List of frame indices to decode.
        Returns:
            result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        """
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        return np.stack([x.to_ndarray(format="rgb24") for x in frames])

    def _extract_features(self, example: VideoExample) -> BatchFeature:
        container = av.open(example.image_path)
        # sample uniformly 8 frames from the video, can sample more for longer videos
        total_frames = container.streams.video[0].frames
        indices = np.arange(0, total_frames, total_frames / 8).astype(int)
        clip = self.read_video_pyav(container, indices)

        inputs = self.processor(text=example.prompt, videos=clip, padding=True, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        for key, values in inputs.items():
            if values.dtype == torch.float32:
                inputs[key] = values.to(self.model.dtype)
        return inputs


class Phi3VideoHfModel(HfModel):
    def _encode_video(self, video_path):
        def uniform_sample(l, n):
            gap = len(l) / n
            idxs = [int(i * gap + gap / 2) for i in range(n)]
            return [l[i] for i in idxs]

        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        sample_fps = round(vr.get_avg_fps() / 1)  # FPS
        frame_idx = [i for i in range(0, len(vr), sample_fps)]
        if len(frame_idx) > MAX_NUM_FRAMES:
            frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
        frames = vr.get_batch(frame_idx).asnumpy()
        frames = [Image.fromarray(v.astype("uint8")).convert("RGB") for v in frames]
        return frames

    def _prep_video(self, example: VideoExample) -> Dict[str, Any]:
        frames = self._encode_video(example.image_path)
        placeholder = ""
        for idx, _ in enumerate(frames, 1):
            placeholder += f"<|image_{idx}|>\n"
        prompt = example.prompt.replace("<|video|>", placeholder)
        return prompt, frames

    def _extract_features(self, example: VideoExample) -> BatchFeature:
        prompt, frames = self._prep_video(example)

        inputs = self.processor(prompt, frames, return_tensors="pt")
        # .to(self.model.device, dtype=self.model.dtype
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        for key, values in inputs.items():
            if values.dtype == torch.float32:
                inputs[key] = values.to(self.model.dtype)
        return inputs
