import numpy as np
from transformers import (
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
from src.config import (
    BERT_BATCH_SIZE,
    BERT_EPOCHS,
    BERT_LR,
    BERT_METRICS_PATH,
    BERT_MODEL_DIR,
    BERT_TEST_PREDICTIONS_PATH,
    BERT_WEIGHT_DECAY,
    RANDOM_SEED,
)
from src.data.load_dataset import load_conll2003
from src.data.preprocess_bert import prepare_bert_datasets
from src.evaluation.metrics import compute_seqeval_metrics
from src.models.bert_ner import build_bert_ner_model
from src.utils import save_json, set_seed

def align_predictions_with_labels(predictions, label_ids, label_names):
    pred_ids = np.argmax(predictions, axis=2)

    true_labels = []
    pred_labels = []

    for pred_row, label_row in zip(pred_ids, label_ids):
        current_true = []
        current_pred = []

        for pred_id, label_id in zip(pred_row, label_row):
            if label_id == -100:
                continue

            current_true.append(label_names[label_id])
            current_pred.append(label_names[pred_id])

        true_labels.append(current_true)
        pred_labels.append(current_pred)

    return true_labels, pred_labels

def build_predictions_payload(raw_test_dataset, true_labels, pred_labels):
    payload = []

    for item, sent_true, sent_pred in zip(raw_test_dataset, true_labels, pred_labels):
        payload.append(
            {
                "tokens": item["tokens"],
                "true_labels": sent_true,
                "predicted_labels": sent_pred,
            }
        )

    return payload

def main():
    set_seed(RANDOM_SEED)

    print("Loading CoNLL-2003 dataset...")
    raw_dataset = load_conll2003()

    print("Preparing BERT tokenizer and tokenized datasets...")
    tokenizer, label_names, tokenized_dataset = prepare_bert_datasets(raw_dataset)

    print("Building BERT token classification model...")
    model, _, _ = build_bert_ner_model(label_names)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        true_labels, pred_labels = align_predictions_with_labels(predictions, labels, label_names)
        metrics = compute_seqeval_metrics(true_labels, pred_labels)
        return {
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        }

    training_args = TrainingArguments(
        output_dir=str(BERT_MODEL_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=BERT_LR,
        per_device_train_batch_size=BERT_BATCH_SIZE,
        per_device_eval_batch_size=BERT_BATCH_SIZE,
        num_train_epochs=BERT_EPOCHS,
        weight_decay=BERT_WEIGHT_DECAY,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=RANDOM_SEED,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Training BERT model...")
    trainer.train()
    
    trainer.save_model(str(BERT_MODEL_DIR))
    tokenizer.save_pretrained(str(BERT_MODEL_DIR))

    print("Evaluating on validation split...")
    val_output = trainer.predict(tokenized_dataset["validation"])
    val_true_labels, val_pred_labels = align_predictions_with_labels(
        val_output.predictions,
        val_output.label_ids,
        label_names,
    )
    val_metrics = compute_seqeval_metrics(val_true_labels, val_pred_labels)

    print("Evaluating on test split...")
    test_output = trainer.predict(tokenized_dataset["test"])
    test_true_labels, test_pred_labels = align_predictions_with_labels(
        test_output.predictions,
        test_output.label_ids,
        label_names,
    )
    test_metrics = compute_seqeval_metrics(test_true_labels, test_pred_labels)

    predictions_payload = build_predictions_payload(
        raw_dataset["test"],
        test_true_labels,
        test_pred_labels,
    )

    metrics_payload = {
        "model": "BERT-based NER",
        "dataset": "CoNLL-2003",
        "validation": val_metrics,
        "test": test_metrics,
    }

    print("Saving predictions and metrics...")
    save_json(predictions_payload, BERT_TEST_PREDICTIONS_PATH)
    save_json(metrics_payload, BERT_METRICS_PATH)

    print("Done.")
    print("Validation F1:", val_metrics["f1"])
    print("Test F1:", test_metrics["f1"])

if __name__ == "__main__":
    main()