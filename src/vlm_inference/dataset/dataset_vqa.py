from typing import Type

from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field

from .dataset_base import VQADataset


class VQAResponse(PydanticBaseModel):
    answer: str = Field(description="Answer for the visual question")


class VizWizVQADataset(VQADataset):
    name = "vizwiz_vqa"
    json_schema: Type[PydanticBaseModel] = VQAResponse
