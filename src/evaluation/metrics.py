from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

def compute_seqeval_metrics(
    y_true: list[list[str]],
    y_pred: list[list[str]],
) -> dict:
    return {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred, digits=4),
    }