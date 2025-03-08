import json
import os
from typing import Callable, Optional
from openai import OpenAI
import pandas as pd
from tqdm.auto import tqdm
from cross_frame.chat import chat
from cross_frame.configuration import ExperimentConfig


import pandas as pd
import numpy as np

from cross_frame.data import load_frames


openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)


def judge_results(reference_config: ExperimentConfig, config: ExperimentConfig):
    reference_model_folder = os.path.join(
        reference_config.pred.prediction_path,
        reference_config.data.test_topic.replace(" ", "_").lower(),
        reference_config.data.train_topic.replace(" ", "_").lower(),
        reference_config.model.model,
    )
    if not os.path.exists(reference_model_folder):
        raise FileNotFoundError(
            f"Reference model folder {reference_model_folder} does not exist"
        )

    output_model_folder = os.path.join(
        config.pred.prediction_path,
        config.data.test_topic.replace(" ", "_").lower(),
        config.data.train_topic.replace(" ", "_").lower(),
        config.model.model,
    )
    if not os.path.exists(output_model_folder):
        raise FileNotFoundError(f"Model folder {output_model_folder} does not exist")

    reference_sample = 0
    reference_eval_path = os.path.join(
        reference_model_folder, f"eval-complete-{reference_sample}.xlsx"
    )
    if not os.path.exists(reference_eval_path):
        raise FileNotFoundError(
            f"Reference eval file {reference_eval_path} does not exist"
        )

    reference_df = pd.read_excel(reference_eval_path)

    if reference_df.isnull().any().any():
        raise ValueError(f"Missing judgments in {reference_eval_path}")

    for sample in tqdm(
        range(config.model.num_samples), total=config.model.num_samples, desc="Samples"
    ):
        eval_complete_path = os.path.join(
            output_model_folder, f"eval-complete-{sample}.xlsx"
        )
        if os.path.exists(eval_complete_path):
            continue
        eval_path = os.path.join(output_model_folder, f"eval-{sample}.xlsx")
        if not os.path.exists(eval_path):
            raise FileNotFoundError(f"Eval file {eval_path} does not exist")

        eval_df = pd.read_excel(eval_path, dtype=str)

        # TODO judge results and save to eval_df
        eval_complete_df = run_judge(reference_df, eval_df, config)

        # save eval_df
        eval_complete_df.to_excel(eval_complete_path, index=False)


def create_few_shot_examples(reference_df: pd.DataFrame) -> list[dict]:
    examples = []
    for idx, row in reference_df.iterrows():
        # first, user message
        frame_text = row["frame"]
        problem_text = row["problems"]
        content = f"""Frame:
{frame_text}
Problems:
{problem_text}"""
        if "sim_f_frame" in row:
            reference_frame_text = row["sim_f_frame"]
            reference_problem_text = row["sim_f_problems"]
            content += f"""
Reference Frame:
{reference_frame_text}
Reference Problems:
{reference_problem_text}"""
        examples.append({"role": "user", "content": content})

        # then, assistant message
        sound = row["Sound"]
        clear = row["Clear"]
        # Yes or No for each
        response = {"Sound": sound, "Clear": clear}
        if "Known" in row:
            known = row["Known"]
            response["Known"] = known
        examples.append({"role": "assistant", "content": json.dumps(response)})
    return examples


def run_judge(
    reference_df: pd.DataFrame,
    df: pd.DataFrame,
    config: ExperimentConfig,
    model: str = "gpt-4o-2024-08-06",
    delay: float | int = 0.1,
) -> pd.DataFrame:
    df = df.copy()
    # create system prompt
    if config.data.test_frames is not None:
        # if we have known reference frames, then they should be used in judgments
        base_messages = [
            {
                "role": "system",
                "content": """You are a helpful assistant that judges the quality of a frame of communication.

    For each frame, you will judge it on three factors:

    - Sound: Does the frame address each of the interpreted problems? Are the interpreted problems correct?
    - Clear: Is the frame clear and easy to understand, and does it articulate a causal interpretation of the problems? Is the frame articulated correctly?
    - Known: Does the frame paraphrase the provided reference frame?

    You will be given a frame and a list of interpreted problems, along with a reference frame and list of interpreted problems for that reference frame.

    You will then judge the frame on the three factors.

    Your response should be in the provided JSON format.""",
            },
        ]
        structured_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "frame_quality_judgment_known",
                "schema": {
                    "type": "object",
                    "properties": {
                        "sound": {
                            "description": "Does the frame address each of the interpreted problems correctly, and are the interpreted problems correct?",
                            "type": "string",
                            "enum": ["Yes", "No"],
                        },
                        "clear": {
                            "description": "Is the frame clear and easy to understand, and does it articulate a causal interpretation of the problems?",
                            "type": "string",
                            "enum": ["Yes", "No"],
                        },
                        "known": {
                            "description": "Does the frame paraphrase the provided reference frame?",
                            "type": "string",
                            "enum": ["Yes", "No"],
                        },
                    },
                    "required": ["sound", "clear", "known"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }
    # first, load reference_df as few-shot examples
    else:
        base_messages = [
            {
                "role": "system",
                "content": """You are a helpful assistant that judges the quality of a frame of communication.

    For each frame, you will judge it on two factors:

    - Sound: Does the frame address each of the interpreted problems? Are the interpreted problems correct?
    - Clear: Is the frame clear and easy to understand, and does it articulate a causal interpretation of the problems? Is the frame articulated correctly?

    You will be given a frame and a list of interpreted problems.

    You will then judge the frame on the two factors.

    Your response should be in the provided JSON format.""",
            },
        ]
        structured_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "frame_quality_judgment",
                "schema": {
                    "type": "object",
                    "properties": {
                        "sound": {
                            "description": "Does the frame address each of the interpreted problems correctly, and are the interpreted problems correct?",
                            "type": "string",
                            "enum": ["Yes", "No"],
                        },
                        "clear": {
                            "description": "Is the frame clear and easy to understand, and does it articulate a causal interpretation of the problems?",
                            "type": "string",
                            "enum": ["Yes", "No"],
                        },
                    },
                    "required": ["sound", "clear"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }
    few_shot_examples = create_few_shot_examples(reference_df)
    base_messages.extend(few_shot_examples)
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Judging"):
        messages = base_messages.copy()
        frame_text = row["frame"]
        problem_text = row["problems"]
        content = f"""Frame:
{frame_text}
Problems:
{problem_text}"""
        if "sim_f_frame" in row:
            reference_frame_text = row["sim_f_frame"]
            reference_problem_text = row["sim_f_problems"]
            content += f"""
Reference Frame:
{reference_frame_text}
Reference Problems:
{reference_problem_text}"""
        messages.append({"role": "user", "content": content})
        # print(messages)
        kwargs = {
            "temperature": 1.0,
            "top_p": 0.4,
            "model": model,
            "seed": 0,
            "max_completion_tokens": 1024,
            "messages": messages,
            "response_format": structured_schema,
        }

        response = chat(openai_client, delay, **kwargs)
        if response is None:
            raise Exception("Safety error, cannot retry")
        response_text = response.choices[0].message.content
        response_schema = json.loads(response_text)
        df.loc[idx, "Sound"] = response_schema["sound"]
        df.loc[idx, "Clear"] = response_schema["clear"]
        if "known" in response_schema:
            df.loc[idx, "Known"] = response_schema["known"]
    return df
