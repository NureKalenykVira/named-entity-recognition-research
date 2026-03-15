from typing import Any

def word2features(sent: list[str], i: int) -> dict[str, Any]:
    word = sent[i]

    features = {
        "bias": 1.0,
        "word.lower()": word.lower(),
        "word[-3:]": word[-3:],
        "word[-2:]": word[-2:],
        "word.isupper()": word.isupper(),
        "word.istitle()": word.istitle(),
        "word.isdigit()": word.isdigit(),
        "word.isalpha()": word.isalpha(),
        "word.has_hyphen": "-" in word,
        "word.length": len(word),
    }

    if i > 0:
        word_prev = sent[i - 1]
        features.update(
            {
                "-1:word.lower()": word_prev.lower(),
                "-1:word.istitle()": word_prev.istitle(),
                "-1:word.isupper()": word_prev.isupper(),
            }
        )
    else:
        features["BOS"] = True

    if i < len(sent) - 1:
        word_next = sent[i + 1]
        features.update(
            {
                "+1:word.lower()": word_next.lower(),
                "+1:word.istitle()": word_next.istitle(),
                "+1:word.isupper()": word_next.isupper(),
            }
        )
    else:
        features["EOS"] = True

    return features

def sent_to_features(tokens: list[str]) -> list[dict[str, Any]]:
    return [word2features(tokens, i) for i in range(len(tokens))]

def sent_to_labels(labels: list[str]) -> list[str]:
    return labels

def prepare_crf_data(sentences: list[dict]) -> tuple[list[list[dict]], list[list[str]]]:
    X = [sent_to_features(sentence["tokens"]) for sentence in sentences]
    y = [sent_to_labels(sentence["labels"]) for sentence in sentences]
    return X, y