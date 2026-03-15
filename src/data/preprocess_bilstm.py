from collections import Counter
from typing import Callable
import torch
from torch.utils.data import Dataset
from src.config import PAD_TOKEN, UNK_TOKEN

def build_token_vocab(sentences: list[dict], min_freq: int = 1) -> dict[str, int]:
    counter = Counter()

    for sentence in sentences:
        counter.update(sentence["tokens"])

    vocab = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }

    for token, freq in counter.items():
        if freq >= min_freq and token not in vocab:
            vocab[token] = len(vocab)

    return vocab

def build_label_vocab(sentences: list[dict]) -> tuple[dict[str, int], dict[int, str]]:
    labels = set()

    for sentence in sentences:
        labels.update(sentence["labels"])

    sorted_labels = sorted(labels)
    label_to_id = {label: idx for idx, label in enumerate(sorted_labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    return label_to_id, id_to_label

def encode_tokens(tokens: list[str], token_vocab: dict[str, int]) -> list[int]:
    unk_id = token_vocab[UNK_TOKEN]
    return [token_vocab.get(token, unk_id) for token in tokens]

def encode_labels(labels: list[str], label_to_id: dict[str, int]) -> list[int]:
    return [label_to_id[label] for label in labels]

class NERDataset(Dataset):
    def __init__(
        self,
        sentences: list[dict],
        token_vocab: dict[str, int],
        label_to_id: dict[str, int],
    ) -> None:
        self.sentences = sentences
        self.token_vocab = token_vocab
        self.label_to_id = label_to_id

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> dict:
        sentence = self.sentences[idx]

        return {
            "tokens": sentence["tokens"],
            "labels": sentence["labels"],
            "input_ids": encode_tokens(sentence["tokens"], self.token_vocab),
            "label_ids": encode_labels(sentence["labels"], self.label_to_id),
        }

def build_collate_fn(pad_token_id: int, pad_label_id: int) -> Callable:
    def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor | list]:
        lengths = [len(item["input_ids"]) for item in batch]
        max_len = max(lengths)

        padded_input_ids = []
        padded_label_ids = []
        attention_masks = []

        for item in batch:
            seq_len = len(item["input_ids"])
            pad_len = max_len - seq_len

            padded_input_ids.append(item["input_ids"] + [pad_token_id] * pad_len)
            padded_label_ids.append(item["label_ids"] + [pad_label_id] * pad_len)
            attention_masks.append([1] * seq_len + [0] * pad_len)

        return {
            "tokens": [item["tokens"] for item in batch],
            "labels": [item["labels"] for item in batch],
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "label_ids": torch.tensor(padded_label_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.uint8),
            "lengths": torch.tensor(lengths, dtype=torch.long),
        }

    return collate_fn