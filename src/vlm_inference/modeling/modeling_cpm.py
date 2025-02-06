import logging
from typing import Any, Callable, Dict, Optional, Tuple, Type

import torch
from decord import VideoReader, cpu
from PIL import Image
from pydantic import BaseModel as PydanticBaseModel

from ..dataset.dataset_base import ImageExample
from ..utils.misc import torch_dtype_from_str
from ..utils.usage_tracking import UsageMetadata
from .modeling_base import VisionLanguageModel

logger = logging.getLogger(__name__)


MAX_NUM_FRAMES = 64  # if cuda OOM set a smaller number


def encode_video(video_path):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype("uint8")).convert("RGB") for v in frames]
    print("num frames:", len(frames))
    return frames


class CpmModel(VisionLanguageModel):

    def __init__(
        self,
        name: str,
        generation_kwargs: Dict[str, Any],
        json_mode: bool,
        dtype: str,
        model_cls: Callable,
        processor_cls: Callable,
        **kwargs
    ):
        super().__init__(name, generation_kwargs, json_mode)

        self.model = model_cls(
            pretrained_model_name_or_path=self.name,
            torch_dtype=torch_dtype_from_str(dtype),
            trust_remote_code=True,
        )

        self.model.to("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = processor_cls(pretrained_model_name_or_path=self.name, trust_remote_code=True)

    def generate(
        self, example: ImageExample, json_schema: Optional[Type[PydanticBaseModel]] = None
    ) -> Tuple[str, UsageMetadata]:

        generated_text = self.model.chat(
            image=Image.open(example.image_path).convert("RGB"),
            msgs=[{"role": "user", "content": example.prompt}],
            tokenizer=self.processor,
            **self.generation_kwargs
        ).strip()

        usage_metadata = UsageMetadata(
            input_token_count=0,
            output_token_count=0,
        )

        return generated_text, usage_metadata


class VideoCpmModel(CpmModel):

    def generate(
        self, example: ImageExample, json_schema: Optional[Type[PydanticBaseModel]] = None
    ) -> Tuple[str, UsageMetadata]:
        frames = encode_video(example.image_path)
        msgs = [
            {"role": "user", "content": frames + [example.prompt]},
        ]

        # Set decode params for video
        params = self.generation_kwargs.copy()
        params["use_image_id"] = False
        params["max_slice_nums"] = 1  # use 1 if cuda OOM and video resolution >  448*448

        generated_text = self.model.chat(image=None, msgs=msgs, tokenizer=self.processor, **params).strip()

        usage_metadata = UsageMetadata(
            input_token_count=0,
            output_token_count=0,
        )

        return generated_text, usage_metadata
