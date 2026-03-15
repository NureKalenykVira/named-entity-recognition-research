from src.config import (
    CRF_METRICS_PATH,
    CRF_MODEL_PATH,
    CRF_TEST_PREDICTIONS_PATH,
    RANDOM_SEED,
)
from src.data.load_dataset import dataset_to_sentences, get_label_names, load_conll2003
from src.data.preprocess_common import convert_sentences_to_label_format
from src.data.preprocess_crf import prepare_crf_data
from src.evaluation.metrics import compute_seqeval_metrics
from src.models.crf_ner import CRFNER
from src.utils import save_json, set_seed

def main() -> None:
    set_seed(RANDOM_SEED)

    print("Loading CoNLL-2003 dataset...")
    dataset = load_conll2003()
    label_names = get_label_names(dataset)

    print("Preparing train/validation/test splits...")
    train_sentences = dataset_to_sentences(dataset["train"])
    validation_sentences = dataset_to_sentences(dataset["validation"])
    test_sentences = dataset_to_sentences(dataset["test"])

    train_data = convert_sentences_to_label_format(train_sentences, label_names)
    validation_data = convert_sentences_to_label_format(validation_sentences, label_names)
    test_data = convert_sentences_to_label_format(test_sentences, label_names)

    X_train, y_train = prepare_crf_data(train_data)
    X_val, y_val = prepare_crf_data(validation_data)
    X_test, y_test = prepare_crf_data(test_data)

    print("Training CRF model...")
    model = CRFNER()
    model.fit(X_train, y_train)

    print("Evaluating on validation split...")
    val_predictions = model.predict(X_val)
    val_metrics = compute_seqeval_metrics(y_val, val_predictions)

    print("Evaluating on test split...")
    test_predictions = model.predict(X_test)
    test_metrics = compute_seqeval_metrics(y_test, test_predictions)

    print("Saving model and outputs...")
    model.save(CRF_MODEL_PATH)

    predictions_payload = []
    for item, pred_labels in zip(test_data, test_predictions):
        predictions_payload.append(
            {
                "tokens": item["tokens"],
                "true_labels": item["labels"],
                "predicted_labels": pred_labels,
            }
        )

    metrics_payload = {
        "model": "CRF",
        "dataset": "CoNLL-2003",
        "validation": val_metrics,
        "test": test_metrics,
    }

    save_json(predictions_payload, CRF_TEST_PREDICTIONS_PATH)
    save_json(metrics_payload, CRF_METRICS_PATH)

    print("Done.")
    print("Validation F1:", val_metrics["f1"])
    print("Test F1:", test_metrics["f1"])

if __name__ == "__main__":
    main()