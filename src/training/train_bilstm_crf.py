import copy
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.config import (
    BILSTM_BATCH_SIZE,
    BILSTM_CRF_METRICS_PATH,
    BILSTM_CRF_MODEL_PATH,
    BILSTM_CRF_TEST_PREDICTIONS_PATH,
    BILSTM_EPOCHS,
    BILSTM_LR,
    DEVICE,
    DROPOUT,
    EMBEDDING_DIM,
    HIDDEN_DIM,
    PAD_TOKEN,
    RANDOM_SEED,
)
from src.data.load_dataset import dataset_to_sentences, get_label_names, load_conll2003
from src.data.preprocess_bilstm import (
    NERDataset,
    build_collate_fn,
    build_label_vocab,
    build_token_vocab,
)
from src.data.preprocess_common import convert_sentences_to_label_format
from src.evaluation.metrics import compute_seqeval_metrics
from src.models.bilstm_crf_ner import BiLSTMCRFNER
from src.utils import save_json, set_seed

def decode_predictions(
    pred_ids: list[list[int]],
    id_to_label: dict[int, str],
) -> list[list[str]]:
    return [[id_to_label[idx] for idx in sentence] for sentence in pred_ids]

def evaluate_model(
    model: BiLSTMCRFNER,
    dataloader: DataLoader,
    id_to_label: dict[int, str],
    device: str,
) -> tuple[dict, list[dict]]:
    model.eval()

    all_true_labels = []
    all_pred_labels = []
    predictions_payload = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            pred_ids = model(input_ids=input_ids, attention_mask=attention_mask)
            pred_labels = decode_predictions(pred_ids, id_to_label)

            true_labels = batch["labels"]
            tokens = batch["tokens"]

            all_true_labels.extend(true_labels)
            all_pred_labels.extend(pred_labels)

            for sentence_tokens, sentence_true, sentence_pred in zip(tokens, true_labels, pred_labels):
                predictions_payload.append(
                    {
                        "tokens": sentence_tokens,
                        "true_labels": sentence_true,
                        "predicted_labels": sentence_pred,
                    }
                )

    metrics = compute_seqeval_metrics(all_true_labels, all_pred_labels)
    return metrics, predictions_payload

def train_one_epoch(
    model: BiLSTMCRFNER,
    dataloader: DataLoader,
    optimizer: Adam,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        label_ids = batch["label_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        optimizer.zero_grad()

        loss = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=label_ids,
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

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

    print("Building vocabularies...")
    token_vocab = build_token_vocab(train_data)
    label_to_id, id_to_label = build_label_vocab(train_data)

    pad_token_id = token_vocab[PAD_TOKEN]
    pad_label_id = label_to_id["O"]

    train_dataset = NERDataset(train_data, token_vocab, label_to_id)
    validation_dataset = NERDataset(validation_data, token_vocab, label_to_id)
    test_dataset = NERDataset(test_data, token_vocab, label_to_id)

    collate_fn = build_collate_fn(pad_token_id=pad_token_id, pad_label_id=pad_label_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BILSTM_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        validation_dataset,
        batch_size=BILSTM_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BILSTM_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    print(f"Using device: {DEVICE}")

    model = BiLSTMCRFNER(
        vocab_size=len(token_vocab),
        num_labels=len(label_to_id),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        pad_token_id=pad_token_id,
        dropout=DROPOUT,
    ).to(DEVICE)

    optimizer = Adam(model.parameters(), lr=BILSTM_LR)

    best_val_f1 = -1.0
    best_model_state = None

    print("Training BiLSTM-CRF model...")
    for epoch in range(BILSTM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)

        val_metrics, _ = evaluate_model(model, val_loader, id_to_label, DEVICE)
        val_f1 = val_metrics["f1"]

        print(
            f"Epoch {epoch + 1}/{BILSTM_EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Validation F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = copy.deepcopy(model.state_dict())

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print("Final evaluation on validation split...")
    val_metrics, _ = evaluate_model(model, val_loader, id_to_label, DEVICE)

    print("Final evaluation on test split...")
    test_metrics, test_predictions_payload = evaluate_model(model, test_loader, id_to_label, DEVICE)

    print("Saving model and outputs...")
    model.save_checkpoint(
        BILSTM_CRF_MODEL_PATH,
        token_vocab=token_vocab,
        label_to_id=label_to_id,
        id_to_label=id_to_label,
    )

    metrics_payload = {
        "model": "BiLSTM-CRF",
        "dataset": "CoNLL-2003",
        "validation": val_metrics,
        "test": test_metrics,
    }

    save_json(test_predictions_payload, BILSTM_CRF_TEST_PREDICTIONS_PATH)
    save_json(metrics_payload, BILSTM_CRF_METRICS_PATH)

    print("Done.")
    print("Validation F1:", val_metrics["f1"])
    print("Test F1:", test_metrics["f1"])


if __name__ == "__main__":
    main()