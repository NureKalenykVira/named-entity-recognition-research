from typing import Iterable

def ids_to_labels(tag_ids: list[int], label_names: list[str]) -> list[str]:
    return [label_names[tag_id] for tag_id in tag_ids]

def convert_sentences_to_label_format(
    sentences: list[dict],
    label_names: list[str],
) -> list[dict]:
    """
    Converts sentence list from tag ids to string labels
    """
    converted = []
    for sentence in sentences:
        converted.append(
            {
                "tokens": sentence["tokens"],
                "labels": ids_to_labels(sentence["ner_tag_ids"], label_names),
            }
        )
    return converted

def flatten(nested: Iterable[Iterable]) -> list:
    return [item for sublist in nested for item in sublist]