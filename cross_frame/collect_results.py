import glob
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

import argparse

import yaml


from cross_frame.configuration import (
    ExperimentConfig,
    ModelConfig,
    DataConfig,
    PredictionConfig,
)

from cross_frame.results import compute_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_folder", type=str, required=True, help="path to folder"
    )
    args = parser.parse_args()
    sample = 0
    results = []
    for config_path in glob.glob(os.path.join(args.config_folder, "*.yaml")):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        config = ExperimentConfig(
            seed=config["seed"],
            model=ModelConfig(**config["model"]),
            data=DataConfig(**config["data"]),
            pred=PredictionConfig(**config["pred"]),
        )

        output_model_folder = os.path.join(
            config.pred.prediction_path,
            config.data.test_topic.replace(" ", "_").lower(),
            config.data.train_topic.replace(" ", "_").lower(),
            config.model.model,
        )

        results_path = os.path.join(output_model_folder, f"results-{sample}.csv")
        if not os.path.exists(results_path):
            compute_results(config)
        # one row, so add to list
        df = pd.read_csv(results_path)
        df["model"] = config.model.model
        df["train_topic"] = config.data.train_topic
        df["test_topic"] = config.data.test_topic
        results.append(df)

    results_df = pd.concat(results)
    # move model to first column
    results_df = results_df[
        ["model", "train_topic", "test_topic", *results_df.columns[:-3]]
    ]
    # sort by F1 score
    results_df = results_df.sort_values(by="F1", ascending=True)
    print(results_df)
    results_df.to_csv(os.path.join(args.config_folder, "results.csv"), index=False)
