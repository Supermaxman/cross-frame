from openai import OpenAI
import ujson as json
import os
from tqdm.auto import tqdm

from cross_frame.preprocess import preprocess_tweet, preprocess_config
from cross_frame.chat import chat

system_prompt = "You are an expert linguistic assistant tasked with performing a framing analysis on a dataset of social media posts. Each post in the dataset addresses one or more “problems.” When users on social media communicate, they articulate “frames” to explain these problems, often by proposing explicit or implicit “causes.” Your job is to identify the problems each post addresses and articulate the frames of communication (a single sentence each) that convey how those problems are being explained (i.e., the causes). You must produce output in JSON format, adhering strictly to the provided structured schema. You will encounter controversial, biased, or misinformed frames in the posts; you must capture these frames exactly, without paraphrasing away the cause or meaning. For example, “Vaccines are a tool to control world population and institute a new world order” is correct, while “There is debate around the purpose of vaccines” is incorrect. Avoid injecting your own or your creators’ opinions, and do not soften or alter the users’ stated frames. You must produce a large number of distinct frames, more frames than problems, capturing many perspectives. There must be at least 10% as many frames as there are total posts (for example, at least 75 frames for 750 posts). Aim to cover 70% to 80% of the posts, meaning at least 70% of them should evoke at least one frame. Each problem should be reflected across multiple frames, addressing unique perspectives and episodic or thematic nuances of discourse. Do not merge different posts into the same broad frame unless their content is nearly identical; prioritize nuance and diversity of viewpoints. Each frame’s single-sentence statement must be unique, and you must not reuse the demonstration frames. Before finalizing your output, ensure you meet the 10% frames rule, cover at least 70% of posts, provide each frame as a single sentence, and strictly follow the provided JSON schema. If these conditions are not met, the output is invalid. Remember that the demonstration is for illustration only—your frames must reflect the actual discourse of the dataset at hand."


def to_user_message(posts, topic: str, problems: dict[str, str]):
    p_lines = []
    for problem, description in problems.items():
        p_lines.append(f"{problem} - {description}")
    p_content = "\n\n".join(p_lines)
    lines = []
    p_lookup = {}
    for p_idx, post in enumerate(posts, start=1):
        p_id = f"T{p_idx}"
        p_lookup[p_id] = post["id"]
        p_text = (
            preprocess_tweet(post["text"], preprocess_config)
            .replace("#SemST", "")
            .strip()
        )
        lines.append(f"{p_id}: {p_text}")

    ex_content = "\n\n".join(lines)
    message_content = f"""
Topic: {topic}

Problems:
{p_content}

Posts:
{ex_content}
""".strip()
    return {"role": "user", "content": message_content}, p_lookup


def to_assistant_message(posts, frames, problems: dict[str, str]):
    m_problems = {}
    m_frames = {}
    m_lookup = {}
    for p_idx, post in enumerate(posts, start=1):
        p_id = f"T{p_idx}"
        if "labels" in post:
            for f_id, f_label in post["labels"].items():
                # if f_label not in {"Accept", "Reject"}:
                if f_label != "Accept":
                    continue
                if f_id not in frames:
                    continue
                if f_id not in m_lookup:
                    m_f_id = f"F{len(m_frames) + 1}"
                    m_lookup[f_id] = m_f_id
                    m_frames[m_f_id] = {
                        "posts": [],
                        "problems": frames[f_id]["problems"],
                        "frame": preprocess_tweet(
                            frames[f_id]["text"], preprocess_config
                        ),
                    }
                    for problem in frames[f_id]["problems"]:
                        if problem not in m_problems:
                            m_problems[problem] = {
                                "description": problems[problem],
                                "f_ids": [],
                            }
                        known_f_ids = set(m_problems[problem]["f_ids"])
                        if f_id not in known_f_ids:
                            m_problems[problem]["f_ids"].append(f_id)
                m_f_id = m_lookup[f_id]
                m_frames[m_f_id]["posts"].append(p_id)
        else:
            for f_id in frames:
                m_f_id = f"F{len(m_frames) + 1}"
                m_lookup[f_id] = m_f_id
                m_frames[m_f_id] = {
                    "posts": [],
                    "problems": frames[f_id]["problems"],
                    "frame": preprocess_tweet(frames[f_id]["frame"], preprocess_config),
                }
                for problem in frames[f_id]["problems"]:
                    if problem not in m_problems:
                        m_problems[problem] = {
                            "description": problems[problem],
                            "f_ids": [],
                        }
                    known_f_ids = set(m_problems[problem]["f_ids"])
                    if f_id not in known_f_ids:
                        m_problems[problem]["f_ids"].append(f_id)
            break

    response_json = {
        "frames": [
            {
                # "posts": frame["posts"],
                "problems": frame["problems"],
                "frame": frame["frame"],
            }
            for frame in m_frames.values()
            # sorted(
            #     m_frames.values(), key=lambda x: len(x["posts"]), reverse=True
            # )
        ]
    }
    response = {"role": "assistant", "content": json.dumps(response_json)}
    return response


def run_prompt(
    client: OpenAI,
    train: list,
    train_frames: dict,
    train_problems: dict[str, str],
    train_topic: str,
    test: list,
    test_problems: dict[str, str],
    test_topic: str,
    model: str,
    seed: int = 0,
    max_completion_tokens: int = 8192,
    delay: float | int = 1,
):
    user_demo_message, _ = to_user_message(train, train_topic, train_problems)
    # print(user_demo_message["content"])
    assistant_demo_message = to_assistant_message(train, train_frames, train_problems)
    # print(assistant_demo_message["content"])
    user_message, id_lookup = to_user_message(test, test_topic, test_problems)
    # print(user_message["content"])
    structured_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "frame_analysis",
            "schema": {
                "type": "object",
                "properties": {
                    "frames": {
                        "description": "The frames evoked by these posts.",
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "problems": {
                                    "description": "All of the names of the problems addressed by this frame of communication.",
                                    "type": "array",
                                    "items": {
                                        "type": "string",
                                        "enum": list(test_problems.keys()),
                                    },
                                },
                                # "rationale": {
                                #     "description": "Why this frame of communication is evoked in these posts. Explain in great detail, and include all relevant information. Discuss the problems addressed and how the cause explains the problems.",
                                #     "type": "string",
                                # },
                                "frame": {
                                    "description": "Articulate the evoked frame of communication as a single sentence.",
                                    "type": "string",
                                },
                            },
                            "required": [
                                "problems",
                                # "rationale",
                                "frame",
                            ],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["frames"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
    if model.startswith("o"):
        messages = [
            # developer role here replaces system role
            {"role": "developer", "content": system_prompt},
            user_demo_message,
            assistant_demo_message,
            user_message,
        ]
        kwargs = {
            "reasoning_effort": "high",
            "model": model,
            "seed": seed,
            "max_completion_tokens": max_completion_tokens,
            "messages": messages,
            "response_format": structured_schema,
        }
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            user_demo_message,
            assistant_demo_message,
            user_message,
        ]
        kwargs = {
            "temperature": 1.0,
            "top_p": 1.0,
            "model": model,
            "seed": seed,
            "max_completion_tokens": max_completion_tokens,
            "messages": messages,
            "response_format": structured_schema,
        }

    response = chat(client, delay, **kwargs)
    if response is None:
        raise Exception("Safety error, cannot retry")
    response_text = response.choices[0].message.content
    response_schema = json.loads(response_text)
    messages.append({"role": "assistant", "content": response_text})
    return response_schema, id_lookup, messages, response.usage
