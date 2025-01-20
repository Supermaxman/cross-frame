from datasets import load_dataset, concatenate_datasets


def load_tweet_eval(name: str):
    dataset = load_dataset("cardiffnlp/tweet_eval", name)
    # test = concatenate_datasets(
    #     [dataset["train"], dataset["validation"], dataset["test"]]
    # )
    test = dataset["test"]
    test = test.filter(lambda x: x["label"] != 0)
    test = test.map(lambda x, idx: {"id": str(idx)}, with_indices=True)
    return test
