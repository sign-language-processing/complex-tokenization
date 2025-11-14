from datasets import load_dataset


def text_dataset(max_samples=None,
                 dataset="Salesforce/wikitext",
                 dataset_config="wikitext-2-raw-v1"):
    dataset = load_dataset(dataset, dataset_config, streaming=True, split="train")
    if max_samples is not None:
        dataset = dataset.take(max_samples)
    return (sample["text"] for sample in dataset)
