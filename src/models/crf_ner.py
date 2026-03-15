from pathlib import Path
import joblib
import sklearn_crfsuite
from src.config import (
    CRF_ALGORITHM,
    CRF_ALL_POSSIBLE_TRANSITIONS,
    CRF_C1,
    CRF_C2,
    CRF_MAX_ITERATIONS,
)

class CRFNER:
    def __init__(self) -> None:
        self.model = sklearn_crfsuite.CRF(
            algorithm=CRF_ALGORITHM,
            c1=CRF_C1,
            c2=CRF_C2,
            max_iterations=CRF_MAX_ITERATIONS,
            all_possible_transitions=CRF_ALL_POSSIBLE_TRANSITIONS,
        )

    def fit(self, X_train: list[list[dict]], y_train: list[list[str]]) -> None:
        self.model.fit(X_train, y_train)

    def predict(self, X: list[list[dict]]) -> list[list[str]]:
        return self.model.predict(X)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: Path) -> None:
        self.model = joblib.load(path)