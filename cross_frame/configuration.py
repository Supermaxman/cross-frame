import dataclasses
from typing import List, Dict, Optional


@dataclasses.dataclass
class ModelConfig:
    num_samples: int
    max_attempts: int
    delay: float
    model: str
    prompt_cost: float
    completion_cost: float
    max_completion_tokens: int


@dataclasses.dataclass
class DataConfig:
    sample_size: int
    train_topic: str
    train_path: str
    train_frames: str
    test_topic: str
    test_path: str
    test_frames: Optional[str] = None


@dataclasses.dataclass
class PredictionConfig:
    prediction_path: str


@dataclasses.dataclass
class ExperimentConfig:
    seed: int
    model: ModelConfig
    data: DataConfig
    pred: PredictionConfig
