import json
import re
from pathlib import Path
from src.config import ENTITY_LABELS

def load_prompt_template(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read()

def build_llm_prompt(template: str, sentence: str) -> str:
    return template.replace("{sentence}", sentence)

def safe_parse_json_array(text: str) -> list[dict]:
    text = text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            return []

    return []

def normalize_entities(entities: list[dict]) -> list[dict]:
    normalized = []

    for item in entities:
        if not isinstance(item, dict):
            continue

        text = item.get("text")
        label = item.get("label")

        if not isinstance(text, str) or not isinstance(label, str):
            continue

        label = label.strip().upper()
        text = text.strip()

        if not text or label not in ENTITY_LABELS:
            continue

        normalized.append(
            {
                "text": text,
                "label": label,
            }
        )

    return normalized

def find_sublist_span(tokens: list[str], entity_tokens: list[str], used_spans: set[tuple[int, int]]) -> tuple[int, int] | None:
    if not entity_tokens or len(entity_tokens) > len(tokens):
        return None

    for start in range(len(tokens) - len(entity_tokens) + 1):
        end = start + len(entity_tokens)
        if tuple(tokens[start:end]) == tuple(entity_tokens):
            span = (start, end)
            if span not in used_spans:
                return span

    return None

def entities_to_iob_labels(tokens: list[str], entities: list[dict]) -> list[str]:
    labels = ["O"] * len(tokens)
    used_spans: set[tuple[int, int]] = set()

    for entity in entities:
        entity_text = entity["text"]
        entity_label = entity["label"]

        entity_tokens = entity_text.split()
        span = find_sublist_span(tokens, entity_tokens, used_spans)

        if span is None:
            continue

        start, end = span
        used_spans.add(span)

        labels[start] = f"B-{entity_label}"
        for idx in range(start + 1, end):
            labels[idx] = f"I-{entity_label}"

    return labels