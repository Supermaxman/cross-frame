from cross_frame.openai_client import run_openai_prompt
from cross_frame.google_client import run_google_prompt, google_models


def run_prompt(
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
    if model in google_models:
        # Google model
        return run_google_prompt(
            train=train,
            train_frames=train_frames,
            train_problems=train_problems,
            train_topic=train_topic,
            test=test,
            test_problems=test_problems,
            test_topic=test_topic,
            model=model,
            seed=seed,
            max_completion_tokens=max_completion_tokens,
            delay=delay,
        )
    else:
        # OpenAI model
        return run_openai_prompt(
            train=train,
            train_frames=train_frames,
            train_problems=train_problems,
            train_topic=train_topic,
            test=test,
            test_problems=test_problems,
            test_topic=test_topic,
            model=model,
            seed=seed,
            max_completion_tokens=max_completion_tokens,
            delay=delay,
        )
