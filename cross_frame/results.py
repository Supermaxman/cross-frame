import os
from typing import Callable, Optional
import pandas as pd
from tqdm.auto import tqdm
from cross_frame.configuration import ExperimentConfig


import pandas as pd
import numpy as np

from cross_frame.data import load_frames


def evaluate(
    frames: set[str],
    unclear: set[str],
    unsound: set[str],
    known_frames: Optional[set[str]] = None,
    known_judged: Optional[set[str]] = None,
):

    total = len(frames)
    sound = len(frames) - len(unsound)
    clear = len(frames) - len(unclear)
    Z = sound / total
    A = clear / total

    metrics = {
        "Z": [Z * 100],
        "A": [A * 100],
    }
    if known_frames is not None and known_judged is not None:
        total_reference = len(known_frames)
        known = len(known_judged)
        # NC /(NC + NF − NK );
        R = clear / (clear + total_reference - known)
        # NK / NF
        Rk = known / total_reference

        F1 = 2.0 * R * Z
        # guard against division by zero
        if F1 > 0:
            F1 = F1 / (R + Z)
        # (NC − NK )/(NT − NK )
        Pa = (clear - known) / (total - known)
        metrics["R"] = [R * 100]
        metrics["Rk"] = [Rk * 100]
        metrics["F1"] = [F1 * 100]
        metrics["Pa"] = [Pa * 100]

    metrics["total"] = [total]
    metrics["sound"] = [sound]
    metrics["clear"] = [clear]
    if known_frames is not None and known_judged is not None:
        metrics["known"] = [known]
        metrics["reference"] = [total_reference]

    return pd.DataFrame(metrics).round(2)


def compute_results(config: ExperimentConfig):
    output_model_folder = os.path.join(
        config.pred.prediction_path,
        config.data.test_topic.replace(" ", "_").lower(),
        config.data.train_topic.replace(" ", "_").lower(),
        config.model.model,
    )
    if not os.path.exists(output_model_folder):
        raise FileNotFoundError(f"Model folder {output_model_folder} does not exist")

    for sample in tqdm(
        range(config.model.num_samples), total=config.model.num_samples, desc="Samples"
    ):
        eval_path = os.path.join(output_model_folder, f"eval-complete-{sample}.xlsx")
        if not os.path.exists(eval_path):
            raise FileNotFoundError(f"Eval file {eval_path} does not exist")

        results_path = os.path.join(output_model_folder, f"results-{sample}.csv")
        if os.path.exists(results_path):
            continue

        # compute metrics
        df = pd.read_excel(eval_path)
        # if any columns are NaN raise an error
        if df.isnull().any().any():
            raise ValueError(f"Missing judgments in {eval_path}")
        unsound = set(df[df["Sound"] == "No"]["f_id"].tolist())
        unclear = set(df[df["Clear"] == "No"]["f_id"].tolist())
        f_ids = set(df["f_id"].tolist())
        known_f_ids = None
        known_judged = None
        if config.data.test_frames is not None:
            known_frames = load_frames(config.data.test_frames)
            known_f_ids = set(known_frames.keys())
            known_judged = set(df[df["Known"] == "Yes"]["f_id"].tolist())

        results = evaluate(f_ids, unclear, unsound, known_f_ids, known_judged)
        print(results)
        results.to_csv(results_path, index=False)
