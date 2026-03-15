from datasets import DatasetDict
from transformers import AutoTokenizer
from src.config import BERT_MAX_LENGTH, BERT_MODEL_NAME

def build_bert_tokenizer():
    return AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

def tokenize_and_align_labels(examples, tokenizer, label_names):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=BERT_MAX_LENGTH,
    )

    aligned_labels = []

    for batch_index, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=batch_index)

        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(labels[word_idx])
            else:
                label_ids.append(-100)

            previous_word_idx = word_idx

        aligned_labels.append(label_ids)

    tokenized_inputs["labels"] = aligned_labels
    return tokenized_inputs

def prepare_bert_datasets(dataset: DatasetDict):
    tokenizer = build_bert_tokenizer()
    label_names = dataset["train"].features["ner_tags"].feature.names

    tokenized_dataset = dataset.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer, label_names),
        batched=True,
    )

    return tokenizer, label_names, tokenized_dataset