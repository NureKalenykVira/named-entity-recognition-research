from pathlib import Path
from tqdm import tqdm
from src.config import (
    LLM_MAX_SENTENCES,
    LLM_METRICS_PATH,
    LLM_TEST_PREDICTIONS_PATH,
    PROJECT_ROOT,
    RANDOM_SEED,
)
from src.data.load_dataset import dataset_to_sentences, get_label_names, load_conll2003
from src.data.preprocess_common import convert_sentences_to_label_format
from src.data.preprocess_llm import (
    build_llm_prompt,
    entities_to_iob_labels,
    load_prompt_template,
    normalize_entities,
    safe_parse_json_array,
)
from src.evaluation.metrics import compute_seqeval_metrics
from src.models.llm_ner import OllamaNERClient
from src.utils import save_json, set_seed

PROMPT_PATH = PROJECT_ROOT / "src" / "prompts" / "llm_ner_prompt.txt"

def prepare_test_subset(test_data: list[dict], max_sentences: int) -> list[dict]:
    return test_data[:max_sentences]

def main() -> None:
    set_seed(RANDOM_SEED)

    print("Loading CoNLL-2003 dataset...")
    dataset = load_conll2003()
    label_names = get_label_names(dataset)

    print("Preparing test split...")
    test_sentences = dataset_to_sentences(dataset["test"])
    test_data = convert_sentences_to_label_format(test_sentences, label_names)
    test_subset = prepare_test_subset(test_data, LLM_MAX_SENTENCES)

    print(f"Using {len(test_subset)} test sentences for LLM inference...")

    print("Loading prompt template...")
    prompt_template = load_prompt_template(PROMPT_PATH)

    print("Initializing Ollama client...")
    client = OllamaNERClient()

    all_true_labels = []
    all_pred_labels = []
    predictions_payload = []

    print("Running LLM inference...")
    for item in tqdm(test_subset, desc="LLM inference"):
        tokens = item["tokens"]
        true_labels = item["labels"]
        sentence = " ".join(tokens)

        prompt = build_llm_prompt(prompt_template, sentence)

        try:
            raw_response = client.generate(prompt)
            parsed_entities = safe_parse_json_array(raw_response)
            normalized_entities = normalize_entities(parsed_entities)
            predicted_labels = entities_to_iob_labels(tokens, normalized_entities)
        except Exception as exc:
            raw_response = f"ERROR: {str(exc)}"
            normalized_entities = []
            predicted_labels = ["O"] * len(tokens)

        all_true_labels.append(true_labels)
        all_pred_labels.append(predicted_labels)

        predictions_payload.append(
            {
                "tokens": tokens,
                "sentence": sentence,
                "true_labels": true_labels,
                "predicted_labels": predicted_labels,
                "raw_llm_response": raw_response,
                "parsed_entities": normalized_entities,
            }
        )

    print("Computing metrics...")
    test_metrics = compute_seqeval_metrics(all_true_labels, all_pred_labels)

    metrics_payload = {
        "model": "LLM-based NER",
        "dataset": "CoNLL-2003",
        "evaluation_subset_size": len(test_subset),
        "test": test_metrics,
    }

    print("Saving predictions and metrics...")
    save_json(predictions_payload, LLM_TEST_PREDICTIONS_PATH)
    save_json(metrics_payload, LLM_METRICS_PATH)

    print("Done.")
    print("Test F1:", test_metrics["f1"])

if __name__ == "__main__":
    main()