import time
from typing import Dict, Any, Tuple

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score



def evaluate_model(name, model, X, y) -> Dict[str, Any]:
    pred = model.predict(X)
    return {
        "model": name,
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
    }



def fit_and_measure(model, X, y) -> Tuple[object, float]:
    start = time.perf_counter()
    model.fit(X, y)
    elapsed = time.perf_counter() - start
    return model, elapsed
