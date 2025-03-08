from dotenv import load_dotenv

load_dotenv()

import argparse

import yaml


from cross_frame.configuration import (
    ExperimentConfig,
    ModelConfig,
    DataConfig,
    PredictionConfig,
)

from cross_frame.judgment import judge_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference", type=str, required=True, help="path to reference config file"
    )
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    args = parser.parse_args()

    with open(args.reference, "r") as f:
        reference_config = yaml.safe_load(f)

    reference_config = ExperimentConfig(
        seed=reference_config["seed"],
        model=ModelConfig(**reference_config["model"]),
        data=DataConfig(**reference_config["data"]),
        pred=PredictionConfig(**reference_config["pred"]),
    )

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    config = ExperimentConfig(
        seed=config["seed"],
        model=ModelConfig(**config["model"]),
        data=DataConfig(**config["data"]),
        pred=PredictionConfig(**config["pred"]),
    )

    judge_results(reference_config, config)
