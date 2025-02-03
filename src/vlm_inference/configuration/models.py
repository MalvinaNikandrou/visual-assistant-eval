from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from omegaconf import MISSING


@dataclass
class Pricing:
    usd_per_input_unit: str
    usd_per_output_unit: str
    unit_tokens: str = "1000000"


@dataclass
class GenerationConfig:
    pass


@dataclass
class APICaptioningGenerationConfig(GenerationConfig):
    max_output_tokens: int = 300
    temperature: float = 0.5
    max_retries: int = 50
    sleep_duration: int = 2


@dataclass
class HfGenerationConfig(GenerationConfig):
    num_return_sequences: int = 1
    do_sample: bool = MISSING
    max_new_tokens: int = MISSING
    temperature: float = MISSING
    top_k: int = MISSING
    max_retries: int = 1


@dataclass
class HfCaptioningGenerationConfig(HfGenerationConfig):
    do_sample: bool = True
    max_new_tokens: int = 300
    temperature: float = 0.5
    top_k: int = 50


@dataclass
class HfVQAGenerationConfig(HfGenerationConfig):
    do_sample: bool = False
    max_new_tokens: int = 75
    temperature: float = 1.0
    top_k: int = 0


@dataclass
class ModelConfig:
    _target_: str = MISSING
    name: str = MISSING
    json_mode: bool = False


@dataclass
class GoogleModelConfig(ModelConfig):
    _target_: str = "vlm_inference.GoogleModel"
    name: str = MISSING
    pricing: Pricing = MISSING


@dataclass
class OpenaiModelConfig(ModelConfig):
    _target_: str = "vlm_inference.OpenaiModel"
    name: str = MISSING
    pricing: Pricing = MISSING


@dataclass
class AnthropicModelConfig(ModelConfig):
    _target_: str = "vlm_inference.AnthropicModel"
    name: str = MISSING
    pricing: Pricing = MISSING


@dataclass
class RekaModelConfig(ModelConfig):
    _target_: str = "vlm_inference.RekaModel"
    name: str = MISSING
    pricing: Pricing = MISSING


@dataclass
class HfModel:
    _target_: str = MISSING
    _partial_: bool = True
    low_cpu_mem_usage: bool = True
    attn_implementation: str = "eager"
    revision: str = "main"
    trust_remote_code: bool = True


@dataclass
class HfProcessor:
    _target_: str = MISSING
    _partial_: bool = True
    use_fast: bool = False
    trust_remote_code: bool = True


@dataclass
class ProcessorConfig:
    _target_: str = MISSING
    _partial_: bool = True


@dataclass
class HfModelConfig(ModelConfig):
    _target_: str = "vlm_inference.HfModel"
    name: str = MISSING
    size: str = MISSING
    dtype: str = MISSING
    model_cls: HfModel = MISSING
    processor_cls: HfProcessor = MISSING
    strip_prompt: bool = False
    postprocess_fn: Optional[ProcessorConfig] = None


@dataclass
class VideoVQAHfModelConfig(HfModelConfig):
    _target_: str = "vlm_inference.VideoHfModel"


@dataclass
class MolmoConfig(HfModelConfig):
    _target_: str = "vlm_inference.MolmoModel"
