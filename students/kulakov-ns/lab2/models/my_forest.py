from typing import Dict, List, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


class RandomForest(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str | float | int | None = "sqrt",
        bootstrap: bool = True,
        criterion: str = "gini",
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.criterion = criterion
        self.random_state = random_state

    def fit(self, X, y):
        X = pd.DataFrame(X).reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)

        self.classes_ = np.sort(y.unique())
        self.n_classes_ = len(self.classes_)
        self.feature_names_in_ = X.columns.to_numpy()
        self.n_features_in_ = X.shape[1]

        self.trees_: List[DecisionTreeClassifier] = []
        self.bootstrap_indices_: List[np.ndarray] = []
        self.oob_indices_: List[np.ndarray] = []

        rng = np.random.RandomState(self.random_state)

        for _ in range(self.n_estimators):
            seed = int(rng.randint(0, 10**9))
            tree = DecisionTreeClassifier(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=seed,
            )

            if self.bootstrap:
                sample_idx = rng.choice(len(X), size=len(X), replace=True)
            else:
                sample_idx = np.arange(len(X))

            used = np.zeros(len(X), dtype=bool)
            used[sample_idx] = True
            oob_idx = np.where(~used)[0]

            tree.fit(X.iloc[sample_idx], y.iloc[sample_idx])

            self.trees_.append(tree)
            self.bootstrap_indices_.append(sample_idx)
            self.oob_indices_.append(oob_idx)

        self.oob_score_ = self._compute_oob_score(X, y)
        self.feature_importances_oob_ = self._compute_oob_feature_importance(X, y)

        return self

    def predict_proba(self, X):
        X = pd.DataFrame(X).reset_index(drop=True)
        proba_sum = np.zeros((len(X), self.n_classes_), dtype=float)

        for tree in self.trees_:
            tree_proba = tree.predict_proba(X)
            aligned = np.zeros((len(X), self.n_classes_), dtype=float)
            for idx, label in enumerate(tree.classes_):
                class_idx = np.where(self.classes_ == label)[0][0]
                aligned[:, class_idx] = tree_proba[:, idx]
            proba_sum += aligned

        return proba_sum / len(self.trees_)

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def score(self, X, y):
        pred = self.predict(X)
        return accuracy_score(y, pred)

    def _compute_oob_score(self, X, y):
        pred, mask = self._collect_oob_predictions(X)
        if not mask.any():
            return np.nan
        y_true = pd.Series(y).reset_index(drop=True).to_numpy()
        return accuracy_score(y_true[mask], pred[mask])

    def _collect_oob_predictions(self, X, permuted_feature=None, permuted_values=None):
        X = pd.DataFrame(X).reset_index(drop=True).copy()
        votes = np.zeros((len(X), self.n_classes_), dtype=float)
        counts = np.zeros(len(X), dtype=int)

        for tree, oob_idx in zip(self.trees_, self.oob_indices_):
            if len(oob_idx) == 0:
                continue

            X_oob = X.iloc[oob_idx].copy()
            if permuted_feature is not None:
                X_oob.loc[:, permuted_feature] = permuted_values[oob_idx]

            tree_proba = tree.predict_proba(X_oob)
            aligned = np.zeros((len(X_oob), self.n_classes_), dtype=float)
            for idx, label in enumerate(tree.classes_):
                class_idx = np.where(self.classes_ == label)[0][0]
                aligned[:, class_idx] = tree_proba[:, idx]

            votes[oob_idx] += aligned
            counts[oob_idx] += 1

        mask = counts > 0
        counts_safe = counts.copy()
        counts_safe[counts_safe == 0] = 1
        probs = votes / counts_safe[:, None]
        pred = self.classes_[np.argmax(probs, axis=1)]
        return pred, mask

    def _compute_oob_feature_importance(self, X, y) -> Dict[str, float]:
        baseline = self.oob_score_
        y_true = pd.Series(y).reset_index(drop=True).to_numpy()
        rng = np.random.RandomState(self.random_state)
        result: Dict[str, float] = {}

        for feature in X.columns:
            permuted_values = X[feature].to_numpy().copy()
            rng.shuffle(permuted_values)
            pred, mask = self._collect_oob_predictions(
                X,
                permuted_feature=feature,
                permuted_values=permuted_values,
            )

            if not mask.any():
                result[feature] = 0.0
                continue

            permuted_score = accuracy_score(y_true[mask], pred[mask])
            result[feature] = baseline - permuted_score

        return result



def oob_scorer(estimator, X, y):
    return estimator.oob_score_



def get_my_forest(X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
    param_grid = {
        "n_estimators": [25, 50, 100],
        "max_depth": [4, 6, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", 0.5],
    }

    train_idx = np.arange(len(X_train))

    grid_search = GridSearchCV(
        estimator=RandomForest(random_state=42),
        param_grid=param_grid,
        scoring=oob_scorer,
        cv=[(train_idx, train_idx)],
        refit=True,
        n_jobs=1,
    )

    return grid_search.fit(X_train, y_train)
