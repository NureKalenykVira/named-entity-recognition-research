import re
from pathlib import Path
import joblib
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer
from src.config import (
    ANALYSIS_DIR,
    BERT_MODEL_DIR,
    BERT_MODEL_NAME,
    BERT_TEST_PREDICTIONS_PATH,
    BILSTM_CRF_MODEL_PATH,
    CRF_MODEL_PATH,
    DEVICE,
    LLM_REQUEST_TIMEOUT,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL_NAME,
    PROJECT_ROOT,
    QUALITATIVE_RESULTS_PATH,
)
from src.data.preprocess_bilstm import build_collate_fn
from src.data.preprocess_crf import sent_to_features
from src.data.preprocess_llm import (
    build_llm_prompt,
    entities_to_iob_labels,
    load_prompt_template,
    normalize_entities,
    safe_parse_json_array,
)
from src.models.bilstm_crf_ner import BiLSTMCRFNER
from src.models.llm_ner import OllamaNERClient
from src.utils import save_json

NEWS_PATH = PROJECT_ROOT / "data" / "sample_texts" / "news_article.txt"
ACADEMIC_PATH = PROJECT_ROOT / "data" / "sample_texts" / "academic_text.txt"
PROMPT_PATH = PROJECT_ROOT / "src" / "prompts" / "llm_ner_prompt.txt"

def read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read().strip()

def simple_sentence_split(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]

def simple_tokenize(sentence: str) -> list[str]:
    return sentence.split()

def load_crf_model():
    return joblib.load(CRF_MODEL_PATH)

def predict_crf(model, tokens: list[str]) -> list[str]:
    X = [sent_to_features(tokens)]
    preds = model.predict(X)
    return list(preds[0])

def load_bilstm_checkpoint():
    checkpoint = torch.load(BILSTM_CRF_MODEL_PATH, map_location=DEVICE)
    token_vocab = checkpoint["token_vocab"]
    label_to_id = checkpoint["label_to_id"]
    id_to_label = checkpoint["id_to_label"]

    pad_token_id = token_vocab["<PAD>"]

    model = BiLSTMCRFNER(
        vocab_size=len(token_vocab),
        num_labels=len(label_to_id),
        embedding_dim=128,
        hidden_dim=256,
        pad_token_id=pad_token_id,
        dropout=0.3,
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, token_vocab, label_to_id, id_to_label

def encode_tokens(tokens: list[str], token_vocab: dict[str, int]) -> list[int]:
    unk_id = token_vocab["<UNK>"]
    return [token_vocab.get(token, unk_id) for token in tokens]

def predict_bilstm(model, token_vocab, id_to_label, tokens: list[str]) -> list[str]:
    input_ids = torch.tensor([encode_tokens(tokens, token_vocab)], dtype=torch.long).to(DEVICE)
    attention_mask = torch.tensor([[1] * len(tokens)], dtype=torch.uint8).to(DEVICE)

    with torch.no_grad():
        pred_ids = model(input_ids=input_ids, attention_mask=attention_mask)

    return [id_to_label[idx] for idx in pred_ids[0]]

def load_bert_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(BERT_MODEL_DIR).to(DEVICE)
    model.eval()
    return tokenizer, model

def predict_bert(tokenizer, model, tokens: list[str]) -> list[str]:
    inputs = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )

    word_ids = inputs.word_ids(batch_index=0)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    pred_ids = outputs.logits.argmax(dim=-1).squeeze(0).tolist()

    labels = []
    previous_word_idx = None

    for pred_id, word_idx in zip(pred_ids, word_ids):
        if word_idx is None:
            continue
        if word_idx != previous_word_idx:
            label = model.config.id2label[pred_id]
            labels.append(label)
        previous_word_idx = word_idx

    return labels

def predict_llm(client: OllamaNERClient, prompt_template: str, tokens: list[str]):
    sentence = " ".join(tokens)
    prompt = build_llm_prompt(prompt_template, sentence)

    try:
        raw_response = client.generate(prompt)
        parsed = safe_parse_json_array(raw_response)
        entities = normalize_entities(parsed)
        labels = entities_to_iob_labels(tokens, entities)
        return labels, raw_response, entities
    except Exception as exc:
        return ["O"] * len(tokens), f"ERROR: {exc}", []

def to_python_list(labels):
    if hasattr(labels, "tolist"):
        return labels.tolist()
    return list(labels)

def run_all_models_on_sentences(sentences: list[str]) -> list[dict]:
    crf_model = load_crf_model()
    bilstm_model, bilstm_token_vocab, _, bilstm_id_to_label = load_bilstm_checkpoint()
    bert_tokenizer, bert_model = load_bert_model_and_tokenizer()
    prompt_template = load_prompt_template(PROMPT_PATH)
    llm_client = OllamaNERClient(
        base_url=OLLAMA_BASE_URL,
        model_name=OLLAMA_MODEL_NAME,
        timeout=LLM_REQUEST_TIMEOUT,
    )

    results = []

    for sentence in sentences:
        tokens = simple_tokenize(sentence)

        if not tokens:
            continue

        crf_labels = predict_crf(crf_model, tokens)
        bilstm_labels = predict_bilstm(bilstm_model, bilstm_token_vocab, bilstm_id_to_label, tokens)
        bert_labels = predict_bert(bert_tokenizer, bert_model, tokens)
        llm_labels, llm_raw_response, llm_entities = predict_llm(llm_client, prompt_template, tokens)

        results.append(
        {
            "sentence": sentence,
            "tokens": list(tokens),
            "crf": to_python_list(crf_labels),
            "bilstm_crf": to_python_list(bilstm_labels),
            "bert": to_python_list(bert_labels),
            "llm": {
                "labels": to_python_list(llm_labels),
                "raw_response": llm_raw_response,
                "parsed_entities": llm_entities,
            },
        }
    )

    return results

def main():
    news_text = read_text(NEWS_PATH)
    academic_text = read_text(ACADEMIC_PATH)

    news_sentences = simple_sentence_split(news_text)
    academic_sentences = simple_sentence_split(academic_text)

    payload = {
        "news_article": run_all_models_on_sentences(news_sentences),
        "academic_text": run_all_models_on_sentences(academic_sentences),
    }

    save_json(payload, QUALITATIVE_RESULTS_PATH)
    print(f"Saved qualitative analysis to: {QUALITATIVE_RESULTS_PATH}")

if __name__ == "__main__":
    main()