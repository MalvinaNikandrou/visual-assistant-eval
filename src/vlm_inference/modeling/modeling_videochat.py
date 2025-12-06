import logging
from typing import Any, Callable, Dict, Optional, Tuple, Type

import torch
from pydantic import BaseModel as PydanticBaseModel

from ..dataset.dataset_base import VideoExample
from ..utils.misc import torch_dtype_from_str
from ..utils.usage_tracking import UsageMetadata
from .modeling_base import VisionLanguageModel

logger = logging.getLogger(__name__)

MAX_NUM_FRAMES = 64


class VideoChatModel(VisionLanguageModel):
    def __init__(
        self,
        name: str,
        generation_kwargs: Dict[str, Any],
        json_mode: bool,
        dtype: str,
        model_cls: Callable,
        processor_cls: Callable,
        **kwargs,
    ):
        super().__init__(name, generation_kwargs, json_mode)

        self.model = model_cls(
            pretrained_model_name_or_path=self.name,
            torch_dtype=torch_dtype_from_str(dtype),  # float16
            trust_remote_code=True,
        )
        mm_llm_compress = False  # use the global compress or not
        if mm_llm_compress:
            self.model.config.mm_llm_compress = True
            self.model.config.llm_compress_type = "uniform0_attention"
            self.model.config.llm_compress_layer_list = [4, 18]
            self.model.config.llm_image_token_ratio_list = [1, 0.75, 0.25]
        else:
            self.model.config.mm_llm_compress = False

        self.model.to("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = processor_cls(pretrained_model_name_or_path=self.name, trust_remote_code=True)

    def generate(
        self,
        example: VideoExample,
        json_schema: Optional[Type[PydanticBaseModel]] = None,
    ) -> Tuple[str, UsageMetadata]:
        generated_text, _ = self.model.chat(
            video_path=example.image_path,
            tokenizer=self.processor,
            user_prompt=example.prompt,
            return_history=True,
            max_num_frames=MAX_NUM_FRAMES,
            generation_config=self.generation_kwargs,
        )
        usage_metadata = UsageMetadata(
            input_token_count=0,
            output_token_count=0,
        )

        return generated_text, usage_metadata
