import json
from pathlib import Path

from src.config import ERROR_ANALYSIS_PATH, QUALITATIVE_RESULTS_PATH
from src.utils import save_json


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def labels_to_spans(tokens: list[str], labels: list[str]) -> list[dict]:
    spans = []
    i = 0

    while i < len(labels):
        label = labels[i]

        if label == "O":
            i += 1
            continue

        if label.startswith("B-"):
            entity_type = label[2:]
            start = i
            entity_tokens = [tokens[i]]
            i += 1

            while i < len(labels) and labels[i] == f"I-{entity_type}":
                entity_tokens.append(tokens[i])
                i += 1

            spans.append(
                {
                    "text": " ".join(entity_tokens),
                    "label": entity_type,
                    "start": start,
                    "end": i,
                }
            )
        else:
            i += 1

    return spans


def build_sentence_summary(item: dict) -> dict:
    tokens = item["tokens"]

    crf_spans = labels_to_spans(tokens, item["crf"])
    bilstm_spans = labels_to_spans(tokens, item["bilstm_crf"])
    bert_spans = labels_to_spans(tokens, item["bert"])
    llm_spans = labels_to_spans(tokens, item["llm"]["labels"])

    return {
        "sentence": item["sentence"],
        "tokens": tokens,
        "crf_labels": item["crf"],
        "bilstm_labels": item["bilstm_crf"],
        "bert_labels": item["bert"],
        "llm_labels": item["llm"]["labels"],
        "crf_spans": crf_spans,
        "bilstm_spans": bilstm_spans,
        "bert_spans": bert_spans,
        "llm_spans": llm_spans,
        "llm_raw_response": item["llm"]["raw_response"],
        "llm_parsed_entities": item["llm"]["parsed_entities"],
    }


def is_llm_empty(item: dict) -> bool:
    return len(item["llm"]["parsed_entities"]) == 0


def bert_differs_from_crf(item: dict) -> bool:
    return item["bert"] != item["crf"]


def bilstm_looks_noisy(item: dict) -> bool:
    labels = item["bilstm_crf"]
    # very simple heuristic: many entity tags in a short sentence
    entity_count = sum(1 for label in labels if label != "O")
    return entity_count >= max(3, len(labels) // 3)


def extract_examples(section_items: list[dict], limit: int = 5) -> dict:
    llm_empty = []
    bert_vs_crf = []
    noisy_bilstm = []

    for item in section_items:
        summary = build_sentence_summary(item)

        if is_llm_empty(item) and len(llm_empty) < limit:
            llm_empty.append(summary)

        if bert_differs_from_crf(item) and len(bert_vs_crf) < limit:
            bert_vs_crf.append(summary)

        if bilstm_looks_noisy(item) and len(noisy_bilstm) < limit:
            noisy_bilstm.append(summary)

    return {
        "llm_empty_examples": llm_empty,
        "bert_vs_crf_examples": bert_vs_crf,
        "bilstm_noisy_examples": noisy_bilstm,
    }


def main() -> None:
    qualitative = load_json(QUALITATIVE_RESULTS_PATH)

    payload = {
        "news_article": extract_examples(qualitative["news_article"], limit=5),
        "academic_text": extract_examples(qualitative["academic_text"], limit=5),
    }

    save_json(payload, ERROR_ANALYSIS_PATH)

    print(f"Error analysis saved to: {ERROR_ANALYSIS_PATH}")


if __name__ == "__main__":
    main()