from datasets import DatasetDict, load_dataset


def load_conll2003() -> DatasetDict:
    """
    Loads the English CoNLL-2003 dataset from Hugging Face datasets.
    """
    dataset = load_dataset("conll2003", trust_remote_code=True)
    return dataset


def get_label_names(dataset: DatasetDict) -> list[str]:
    """
    Extracts label names from dataset features.
    """
    return dataset["train"].features["ner_tags"].feature.names


def dataset_to_sentences(split) -> list[dict]:
    """
    Converts a dataset split into a list of sentence dictionaries:
    {
        "tokens": [...],
        "ner_tag_ids": [...],
    }
    """
    sentences = []
    for item in split:
        sentences.append(
            {
                "tokens": item["tokens"],
                "ner_tag_ids": item["ner_tags"],
            }
        )
    return sentences