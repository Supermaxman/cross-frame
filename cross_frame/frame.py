import os
import argparse
from dotenv import load_dotenv

from openai import OpenAI
import yaml

from cross_frame.configuration import (
    ExperimentConfig,
    ModelConfig,
    DataConfig,
    PredictionConfig,
)
from cross_frame.experiment import run_experiment


def run(config: ExperimentConfig):
    load_dotenv()

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    run_experiment(client, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    config = ExperimentConfig(
        seed=config["seed"],
        model=ModelConfig(**config["model"]),
        data=DataConfig(**config["data"]),
        pred=PredictionConfig(**config["pred"]),
    )
    run(config)
