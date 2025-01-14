import logging
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Type

import hydra
from jinja2 import Template
from pydantic import BaseModel as PydanticBaseModel

from ..utils.json_parsing import parse_pydantic_schema
from abc import ABC, abstractmethod
import json
logger = logging.getLogger(__name__)


class BaseDataset(ABC):
    name: str
    json_schema: Type[PydanticBaseModel]

    def __init__(self, path: str, template_name: str, **kwargs):
        self._load_dataset(Path(path))
        self._load_template(template_name)

    @abstractmethod
    def _load_dataset(self, data_dir: Path) -> None:
        pass

    def _load_template(self, template_name: str) -> None:
        template_path = Path(hydra.utils.get_original_cwd()) / "templates" / f"{template_name}.txt"
        with open(template_path) as f:
            self.template: Template = Template(f.read())

    @cache
    def get_prompt(self) -> str:
        return self.template.render(json_schema=parse_pydantic_schema(self.json_schema))

    @abstractmethod
    def __getitem__(self, index: int):
        pass

    def __len__(self) -> int:
        return len(self.data)


@dataclass
class ImageExample:
    image_path: str
    prompt: str
    ground_truth: list[str] = None


class ImageDataset(BaseDataset):

    def _load_dataset(self, data_dir: Path) -> None:
        if not data_dir.exists() and not data_dir.is_dir():
            raise FileNotFoundError(f"Directory `{data_dir}` does not exist.")

        self.data = [str(image_path.resolve()) for image_path in data_dir.rglob("*.jpg")]
        logger.info(f"Resolved {len(self.data)} images in directory `{data_dir}`.")

    def __getitem__(self, index: int) -> ImageExample:
        image_path = self.data[index]
        prompt = self.get_prompt()

        return ImageExample(image_path=image_path, prompt=prompt)


class VQADataset(BaseDataset):
    def __init__(self, images_path: str, path:str, template_name: str):
        self.images_path = Path(images_path)
        super().__init__(path, template_name)
        
    def _resolve_image_path(self, example) -> str:
        example["image_path"] = str(self.images_path.joinpath(example["image"]).resolve())
        return example
 
    def _load_dataset(self, annotations_path: str) -> None:
        with open(annotations_path) as f:
            self.data = [self._resolve_image_path(example) for example in json.load(f)]

    @cache
    def get_prompt(self, question: str) -> str:
        return self.template.render(question=question, json_schema=parse_pydantic_schema(self.json_schema))

    def __getitem__(self, index: int) -> ImageExample:
        example = self.data[index]
        image_path = example["image_path"]
        prompt = self.get_prompt(example["question"])
        answers = example.get("answers", None)
        if answers is not None:
            ground_truth = [answer["answer"] for answer in answers]
        return ImageExample(image_path=image_path, prompt=prompt, ground_truth=ground_truth)

    def __len__(self) -> int:
        return len(self.data)