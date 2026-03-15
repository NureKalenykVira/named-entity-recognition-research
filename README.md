# Named Entity Recognition Research

This repository contains the code and experimental materials for a research project on **named entity recognition (NER)**.

The work focuses on the implementation, evaluation, and comparison of four approaches to NER:

- **CRF**
- **BiLSTM-CRF**
- **BERT-based NER**
- **LLM-based NER**

## Research goal

The goal of the project is to compare classical, neural, transformer-based, and large language model approaches to named entity recognition under a common experimental setup.

The experiments were conducted on the **CoNLL-2003** dataset using shared evaluation metrics:
- **precision**
- **recall**
- **F1-score**

## Project structure

```text
.
├── src/                  # source code for training, inference, and evaluation
├── data/                 # dataset-related files (if included)
├── outputs/              # saved metrics, predictions, comparison tables, qualitative analysis
├── requirements.txt      # Python dependencies
└── README.md             # project description
````

## Implemented models

### 1. CRF

A conditional random field model based on manually designed token-level features.

### 2. BiLSTM-CRF

A neural sequence labeling model built with embedding, bidirectional LSTM, linear layer, and CRF decoding.

### 3. BERT-based NER

A transformer-based token classification model using `bert-base-cased`.

### 4. LLM-based NER

A prompt-based approach using a local LLM through **Ollama** with post-processing into BIO labels.

## Dataset

The experiments use the **CoNLL-2003** dataset.

The English portion of the dataset is used, with the standard split:

* **train** — 14,041 sentences
* **validation** — 3,250 sentences
* **test** — 3,453 sentences

The dataset contains four entity types:

* `PER`
* `ORG`
* `LOC`
* `MISC`

The labels are represented in the **BIO** format.

## Evaluation

All models were evaluated using:

* **precision**
* **recall**
* **F1-score**

Evaluation was performed at the entity level based on BIO-tag sequences.

## Main results

Among the compared approaches, **BERT-based NER** achieved the best overall performance in the conducted experiments.

The repository also includes:

* saved metric files for each model
* comparison tables
* prediction outputs
* qualitative analysis on external news and academic texts
* selected error analysis examples

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Notes

The file below is not included in the repository because of its size:

```text
outputs/models/bert/model.safetensors
```

If needed, the model can be re-trained using the provided training scripts.
