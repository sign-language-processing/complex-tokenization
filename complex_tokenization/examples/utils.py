import argparse

from datasets import load_dataset


def text_dataset(max_samples=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--dataset_config", default="wikitext-2-raw-v1")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, args.dataset_config, streaming=True, split="train")
    if max_samples is not None:
        dataset = dataset.take(max_samples)
    return (sample["text"] for sample in dataset)
