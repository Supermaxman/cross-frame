import os
from typing import Callable

from tqdm.auto import tqdm
from cross_frame.configuration import ExperimentConfig
from cross_frame.data import (
    load_frames,
    load_predictions,
)
from cross_frame.excel_utils import create_excel
from sentence_transformers import SentenceTransformer


def load_sim_model(known_frames: dict) -> Callable[[str], tuple[str, dict]]:
    model_name = "all-MiniLM-L6-v2"
    sim_model = SentenceTransformer(model_name)
    f_texts = [f["text"] if "text" in f else f["frame"] for f in known_frames.values()]
    f_embeddings = sim_model.encode(f_texts)

    def run_model(frame_text: str) -> tuple[str, dict]:
        frame_embedding = sim_model.encode(frame_text)
        sim_scores = sim_model.similarity(frame_embedding, f_embeddings)
        k_f_id = list(known_frames.keys())[sim_scores.argmax()]
        k_f = known_frames[k_f_id]
        return k_f_id, k_f

    return run_model


def create_eval(config: ExperimentConfig):
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
        output_path = os.path.join(output_model_folder, f"predictions-{sample}.json")
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Predictions file {output_path} does not exist")

        # {
        #     "seed": sample,
        #     "attempt": attempt,
        #     "attempt_seed": attempt_seed,
        #     "response_schema": response_schema,
        #     "id_lookup": id_lookup,
        # }
        predictions = load_predictions(output_path)
        known_frames = None
        sim_model = None
        if config.data.test_frames is not None:
            known_frames = load_frames(config.data.test_frames)
            sim_model = load_sim_model(known_frames)
        response_schema = predictions["response_schema"]
        frames = response_schema["frames"]
        rows = []
        for idx, frame in tqdm(
            enumerate(frames, start=1), total=len(frames), desc="Frames"
        ):
            frame_text = frame["frame"]
            row = {
                "f_id": f"F{idx}",
                "frame": frame_text,
            }
            problem_list = []
            for problem in frame["problems"]:
                # print(problem)
                problem_list.append(problem)
            row["problems"] = "\n".join(problem_list)
            if known_frames is not None:
                # TODO find most similar frame to frame_text from known frames
                k_f_id, k_f = sim_model(frame_text)
                row["sim_f_id"] = k_f_id
                row["sim_f_frame"] = k_f["text"] if "text" in k_f else k_f["frame"]
                row["sim_f_problems"] = "\n".join(k_f["problems"])
            rows.append(row)
        labels = {
            "Sound": {
                "values": ["Yes", "No"],
                "message": "Does the frame of communication address each of the interpreted problems?",
            },
            "Clear": {
                "values": ["Yes", "No"],
                "message": "Is the frame of communication clear and easy to understand, and does it articulate a causal interpretation of the problems?",
            },
        }
        if known_frames is not None:
            labels["Known"] = {
                "values": ["Yes", "No"],
                "message": "Does the frame of communication paraphrase a known frame?",
            }
        columns = ["f_id", "frame", "problems"]
        if known_frames is not None:
            columns.extend(["sim_f_id", "sim_f_frame", "sim_f_problems"])
        create_excel(
            data=rows,
            output_path=os.path.join(output_model_folder, f"eval-{sample}.xlsx"),
            columns=columns,
            labels=labels,
        )
