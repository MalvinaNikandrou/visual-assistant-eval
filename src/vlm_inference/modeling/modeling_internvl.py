import logging
from typing import Any, Callable, Dict, Optional, Tuple, Type

import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from pydantic import BaseModel as PydanticBaseModel
from torchvision.transforms.functional import InterpolationMode

from ..dataset.dataset_base import ImageExample
from ..utils.misc import torch_dtype_from_str
from ..utils.usage_tracking import UsageMetadata
from .modeling_base import VisionLanguageModel

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


# video multi-round conversation (视频多轮对话)
def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array(
        [int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_segments)]
    )
    return frame_indices


def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


class InternVLModel(VisionLanguageModel):
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

        self.dtype = torch_dtype_from_str(dtype)
        self.model = model_cls(
            pretrained_model_name_or_path=self.name,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        )

        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.processor = processor_cls(pretrained_model_name_or_path=self.name, trust_remote_code=True)

    def perpare_input(self, example: ImageExample) -> Any:
        pixel_values = load_image(example.image_path, max_num=12).to(self.dtype).to(self.model.device)
        text = f"<image>\n{example.prompt}"
        return pixel_values, text

    def generate(
        self,
        example: ImageExample,
        json_schema: Optional[Type[PydanticBaseModel]] = None,
    ) -> Tuple[str, UsageMetadata]:
        pixel_values, text = self.perpare_input(example)
        generated_text = self.model.chat(self.processor, pixel_values, text, self.generation_kwargs).strip()

        usage_metadata = UsageMetadata(
            input_token_count=0,
            output_token_count=0,
        )

        return generated_text, usage_metadata


class VideoInternVLModel(InternVLModel):
    def perpare_input(self, example: ImageExample) -> Any:
        pixel_values, num_patches_list = load_video(example.image_path, num_segments=8, max_num=1)
        pixel_values = pixel_values.to(self.dtype).to(self.model.device)
        video_prefix = "".join([f"Frame{i + 1}: <image>\n" for i in range(len(num_patches_list))])
        text = f"{video_prefix}{example.prompt}"
        return pixel_values, num_patches_list, text

    def generate(
        self,
        example: ImageExample,
        json_schema: Optional[Type[PydanticBaseModel]] = None,
    ) -> Tuple[str, UsageMetadata]:
        pixel_values, num_patches_list, text = self.perpare_input(example)
        generated_text = self.model.chat(
            self.processor,
            pixel_values,
            text,
            self.generation_kwargs,
            num_patches_list=num_patches_list,
            history=None,
            return_history=False,
        )

        usage_metadata = UsageMetadata(
            input_token_count=0,
            output_token_count=0,
        )

        return generated_text, usage_metadata
