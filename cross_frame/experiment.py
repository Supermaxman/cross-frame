import os

from tqdm.auto import tqdm
from cross_frame.chat import print_messages
from cross_frame.configuration import ExperimentConfig
from cross_frame.costs import usage_to_text
from cross_frame.data import (
    load_test,
    load_train,
    sample_test,
    sample_train,
    save_predictions,
)
from cross_frame.prompting import run_prompt
from openai import OpenAI

from cross_frame.rejection import check_results, stats_to_text


def run_experiment(client: OpenAI, config: ExperimentConfig):
    # output folder should be pred_path/test_topic/train_topic/model
    output_model_folder = os.path.join(
        config.pred.prediction_path,
        config.data.test_topic.replace(" ", "_").lower(),
        config.data.train_topic.replace(" ", "_").lower(),
        config.model.model,
    )
    os.makedirs(output_model_folder, exist_ok=True)

    for sample in tqdm(range(config.model.num_samples), total=config.model.num_samples):
        output_path = os.path.join(output_model_folder, f"predictions-{sample}.json")
        if os.path.exists(output_path):
            continue
        results = None
        for attempt in range(config.model.max_attempts):
            try:
                results = run_attempt(client, config, sample, attempt)
            except Exception as e:
                print(f"s={sample}, a={attempt} failed with error: {e}")
            if results is not None:
                break
        if results is None:
            print(f"s={sample} failed after {config.model.max_attempts} attempts")
            continue

        save_predictions(results, output_path)


def run_attempt(
    client: OpenAI, config: ExperimentConfig, sample: int = 0, attempt: int = 0
) -> dict | None:
    attempt_seed = config.seed + 100 * sample + 1000 * attempt
    train, train_problems, train_frames_all = load_train(
        config.data.train_topic, config.data.train_path, config.data.train_frames
    )
    train_shuffled, train_frames = sample_train(
        train, train_frames_all, config.data.sample_size, attempt_seed
    )
    print(
        f"s={sample}, a={attempt} Train size: {len(train_shuffled)}, {len(train_frames)} frames"
    )
    test, test_problems = load_test(config.data.test_topic, config.data.test_path)
    test_shuffled = sample_test(test, config.data.sample_size, attempt_seed)
    print(f"s={sample}, a={attempt} Test size: {len(test_shuffled)}")

    response_schema, id_lookup, messages, usage = run_prompt(
        client=client,
        train=train_shuffled,
        train_frames=train_frames,
        train_problems=train_problems,
        train_topic=config.data.train_topic,
        test=test_shuffled,
        test_problems=test_problems,
        test_topic=config.data.test_topic,
        model=config.model.model,
        seed=attempt_seed,
        max_completion_tokens=config.model.max_completion_tokens,
        delay=config.model.delay,
    )
    print(
        f"s={sample}, a={attempt} Test size: {len(test_shuffled)}, {len(response_schema['frames'])} frames"
    )
    print(usage_to_text(usage))
    good, stats = check_results(response_schema, test_shuffled, test_problems)
    print(stats_to_text(stats))

    if not good:
        print(f"s={sample}, a={attempt} failed")
        print_messages(messages)
        # print_results(response_schema)
        # follow_up = input()
        # resp = run_follow_up(
        #     messages=messages + [{"role": "user", "content": follow_up}],
        #     seed=s,
        #     max_completion_tokens=16384,
        #     model=model,
        # )
        # print(resp)
        return None
    # print_results(response_schema)
    results = {
        "seed": sample,
        "attempt": attempt,
        "attempt_seed": attempt_seed,
        "response_schema": response_schema,
        "id_lookup": id_lookup,
    }
    return results
