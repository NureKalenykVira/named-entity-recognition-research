from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_TEXTS_DIR = DATA_DIR / "sample_texts"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
METRICS_DIR = OUTPUTS_DIR / "metrics"
ANALYSIS_DIR = OUTPUTS_DIR / "analysis"

CRF_MODEL_DIR = MODELS_DIR / "crf"
BILSTM_CRF_MODEL_DIR = MODELS_DIR / "bilstm_crf"
BERT_MODEL_DIR = MODELS_DIR / "bert"
LLM_MODEL_DIR = MODELS_DIR / "llm"

DATASET_NAME = "conll2003"

ENTITY_LABELS = ["PER", "ORG", "LOC", "MISC"]

RANDOM_SEED = 42

CRF_ALGORITHM = "lbfgs"
CRF_C1 = 0.1
CRF_C2 = 0.1
CRF_MAX_ITERATIONS = 100
CRF_ALL_POSSIBLE_TRANSITIONS = True

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

EMBEDDING_DIM = 128
HIDDEN_DIM = 256
DROPOUT = 0.3

BILSTM_BATCH_SIZE = 32
BILSTM_EPOCHS = 8
BILSTM_LR = 0.001

BERT_MODEL_NAME = "bert-base-cased"
BERT_MAX_LENGTH = 128
BERT_BATCH_SIZE = 16
BERT_EPOCHS = 3
BERT_LR = 2e-5
BERT_WEIGHT_DECAY = 0.01

OLLAMA_BASE_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL_NAME = "qwen2.5:3b-instruct"
LLM_MAX_SENTENCES = 300
LLM_REQUEST_TIMEOUT = 120
LLM_TEMPERATURE = 0.0

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"

CRF_MODEL_PATH = CRF_MODEL_DIR / "crf_model.joblib"
CRF_TEST_PREDICTIONS_PATH = PREDICTIONS_DIR / "crf_test_predictions.json"
CRF_METRICS_PATH = METRICS_DIR / "crf_metrics.json"

BILSTM_CRF_MODEL_PATH = BILSTM_CRF_MODEL_DIR / "bilstm_crf_model.pt"
BILSTM_CRF_TEST_PREDICTIONS_PATH = PREDICTIONS_DIR / "bilstm_crf_test_predictions.json"
BILSTM_CRF_METRICS_PATH = METRICS_DIR / "bilstm_crf_metrics.json"

BERT_TEST_PREDICTIONS_PATH = PREDICTIONS_DIR / "bert_test_predictions.json"
BERT_METRICS_PATH = METRICS_DIR / "bert_metrics.json"

LLM_TEST_PREDICTIONS_PATH = PREDICTIONS_DIR / "llm_test_predictions.json"
LLM_METRICS_PATH = METRICS_DIR / "llm_metrics.json"

QUALITATIVE_RESULTS_PATH = ANALYSIS_DIR / "qualitative_analysis.json"

COMPARISON_TABLE_CSV_PATH = METRICS_DIR / "comparison_table.csv"
COMPARISON_TABLE_JSON_PATH = METRICS_DIR / "comparison_table.json"
ERROR_ANALYSIS_PATH = ANALYSIS_DIR / "error_analysis.json"