import copy
import logging
import numpy as np
from typing import Any, Dict, Optional, Tuple, Type
import warnings

import torch
from decord import VideoReader, cpu
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from pydantic import BaseModel as PydanticBaseModel

from ..dataset.dataset_base import ImageExample
from ..utils.misc import torch_dtype_from_str
from ..utils.usage_tracking import UsageMetadata
from .modeling_base import VisionLanguageModel


warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


def load_video(video_path, max_frames_num, fps=1, force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps() / fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i / fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i / vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames, frame_time, video_time


class LlaVAVideoModel(VisionLanguageModel):

    def __init__(
        self,
        name: str,
        generation_kwargs: Dict[str, Any],
        json_mode: bool,
        dtype: str,
        model_name: str,
        conv_template: str,
        max_frames_num: int = 64,
        **kwargs,
    ):
        super().__init__(name, generation_kwargs, json_mode)

        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(
            self.name,
            None,
            model_name,
            torch_dtype=dtype,
            device_map="auto",
        )
        self.max_frames_num = max_frames_num
        self.conv_template = conv_template

    def _load_video(self, video_path):
        video, frame_time, video_time = load_video(video_path, self.max_frames_num, 1, force_sample=True)
        video = self.image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().half()
        video = [video]
        return video, frame_time, video_time

    def _prepare_prompt(self, prompt, video, video_time, frame_time):
        time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}. Please answer the following questions related to this video."
        question = DEFAULT_IMAGE_TOKEN + f"\n{time_instruciton}\n{prompt}"
        conv = copy.deepcopy(conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        return prompt_question

    def generate(
        self, example: ImageExample, json_schema: Optional[Type[PydanticBaseModel]] = None
    ) -> Tuple[str, UsageMetadata]:
        video, frame_time, video_time = self._load_video(example.image_path)
        prompt = self._prepare_prompt(example.prompt, video, video_time, frame_time)
        inputs = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(self.model.device)
        )
        cont = self.model.generate(
            inputs,
            images=video,
            modalities=["video"],
            **self.generation_kwargs,
        )
        generated_text = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()

        usage_metadata = UsageMetadata(
            input_token_count=0,
            output_token_count=0,
        )

        return generated_text, usage_metadata
