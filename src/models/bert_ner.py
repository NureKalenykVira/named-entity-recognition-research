from transformers import AutoModelForTokenClassification
from src.config import BERT_MODEL_NAME

def build_bert_ner_model(label_names: list[str]):
    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {label: i for i, label in enumerate(label_names)}

    model = AutoModelForTokenClassification.from_pretrained(
        BERT_MODEL_NAME,
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
    )

    return model, id2label, label2id