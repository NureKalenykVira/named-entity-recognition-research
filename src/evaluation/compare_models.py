import json
from pathlib import Path

import pandas as pd

from src.config import (
    BERT_METRICS_PATH,
    BILSTM_CRF_METRICS_PATH,
    COMPARISON_TABLE_CSV_PATH,
    COMPARISON_TABLE_JSON_PATH,
    CRF_METRICS_PATH,
    LLM_METRICS_PATH,
)
from src.utils import ensure_dir, save_json


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_row(metrics_payload: dict) -> dict:
    model_name = metrics_payload["model"]
    dataset_name = metrics_payload["dataset"]

    row = {
        "model": model_name,
        "dataset": dataset_name,
        "validation_precision": None,
        "validation_recall": None,
        "validation_f1": None,
        "test_precision": None,
        "test_recall": None,
        "test_f1": None,
        "evaluation_subset_size": metrics_payload.get("evaluation_subset_size"),
    }

    if "validation" in metrics_payload:
        row["validation_precision"] = metrics_payload["validation"].get("precision")
        row["validation_recall"] = metrics_payload["validation"].get("recall")
        row["validation_f1"] = metrics_payload["validation"].get("f1")

    if "test" in metrics_payload:
        row["test_precision"] = metrics_payload["test"].get("precision")
        row["test_recall"] = metrics_payload["test"].get("recall")
        row["test_f1"] = metrics_payload["test"].get("f1")

    return row


def main() -> None:
    metric_files = [
        CRF_METRICS_PATH,
        BILSTM_CRF_METRICS_PATH,
        BERT_METRICS_PATH,
        LLM_METRICS_PATH,
    ]

    rows = []
    for path in metric_files:
        payload = load_json(path)
        rows.append(extract_row(payload))

    df = pd.DataFrame(rows)

    # Optional: order by test_f1 descending
    df = df.sort_values(by="test_f1", ascending=False, na_position="last").reset_index(drop=True)

    ensure_dir(COMPARISON_TABLE_CSV_PATH.parent)
    df.to_csv(COMPARISON_TABLE_CSV_PATH, index=False, encoding="utf-8-sig")
    save_json(df.to_dict(orient="records"), COMPARISON_TABLE_JSON_PATH)

    print("Comparison table saved:")
    print(f"CSV:  {COMPARISON_TABLE_CSV_PATH}")
    print(f"JSON: {COMPARISON_TABLE_JSON_PATH}")
    print()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()