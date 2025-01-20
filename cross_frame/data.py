import random
import ujson as json

from cross_frame.immigration import load_immigration
from cross_frame.covid import load_covid

topic_loaders = {
    "COVID-19 Vaccines": load_covid,
    "Immigration": load_immigration,
}


def sample_train(data: list, frames: dict, sample_size: int, seed: int = 0):
    # first, copy data list so we don't modify it
    new_data = data.copy()
    random.seed(seed)
    random.shuffle(new_data)
    new_data = new_data[:sample_size]
    keep_f_ids = set()
    for ex in new_data:
        for f_id, f_label in ex["labels"].items():
            if f_label == "Accept":
                keep_f_ids.add(f_id)

    new_frames = {f_id: f for f_id, f in frames.items() if f_id in keep_f_ids}
    return new_data, new_frames


def sample_test(data: list, sample_size: int, seed: int = 0):
    random.seed(seed)
    random.shuffle(data)
    return data[:sample_size]


def load_frames(file_path: str):
    with open(file_path, "r") as file:
        frames = json.load(file)
    return frames


def load_train(
    topic: str,
    file_path: str,
    frames_path: str,
):
    data, problems = topic_loaders[topic](file_path)
    frames = load_frames(frames_path)
    return data, problems, frames


def load_test(topic: str, file_path: str):
    data, problems = topic_loaders[topic](file_path)
    return data, problems


def save_predictions(predictions: list | dict, output_path: str):
    with open(output_path, "w") as f:
        json.dump(predictions, f)


def load_predictions(output_path: str):
    with open(output_path, "r") as f:
        return json.load(f)
