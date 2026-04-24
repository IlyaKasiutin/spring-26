import numpy as np
import time
from collections import Counter

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier as SklearnRF
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score


class RandomForest(BaseEstimator, ClassifierMixin): #  тут мы наследуемся чтобы иметь потом совместимость с GridSearchCV
    def __init__(
        self,
        n_estimators=100,
        max_features="sqrt",
        max_depth=None,
        min_samples_split=2,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state

    def _max_features(self, n_features):
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if self.max_features == "log2":
            return max(1, int(np.log2(n_features)))
        if isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        if isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        return n_features

    def _bootstrap(self, n, rng):
        idx = rng.integers(0, n, size=n)
        oob = np.setdiff1d(np.arange(n), idx)
        return idx, oob

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        n, m = X.shape

        self.classes_ = np.unique(y)
        class_idx = {c: i for i, c in enumerate(self.classes_)}

        self.estimators_ = []
        self.oob_indices_ = []

        votes = np.zeros((n, len(self.classes_)))
        max_feat = self._max_features(m)

        for _ in range(self.n_estimators):
            bag, oob = self._bootstrap(n, rng)

            tree = DecisionTreeClassifier(
                max_features=max_feat,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=int(rng.integers(0, 2**31)),
            )
            tree.fit(X[bag], y[bag])

            self.estimators_.append(tree)
            self.oob_indices_.append(oob)

            if len(oob):
                preds = tree.predict(X[oob])
                for i, p in zip(oob, preds):
                    votes[i, class_idx[p]] += 1

        mask = votes.sum(axis=1) > 0
        if mask.any():
            pred = self.classes_[votes[mask].argmax(axis=1)]
            self.oob_score_ = accuracy_score(y[mask], pred)
        else:
            self.oob_score_ = None

        return self

    def predict(self, X):
        preds = np.array([t.predict(X) for t in self.estimators_])
        return np.array([Counter(col).most_common(1)[0][0] for col in preds.T])

    def oob_feature_importance(self, X, y):
        rng = np.random.default_rng(self.random_state)
        m = X.shape[1]
        imp = np.zeros(m)

        for tree, oob in zip(self.estimators_, self.oob_indices_):
            if not len(oob):
                continue

            Xo, yo = X[oob], y[oob]
            base = accuracy_score(yo, tree.predict(Xo))

            for j in range(m):
                Xp = Xo.copy()
                Xp[:, j] = Xp[rng.permutation(len(oob)), j]
                acc = accuracy_score(yo, tree.predict(Xp))
                imp[j] += base - acc

        imp /= self.n_estimators
        imp = np.clip(imp, 0, None)
        s = imp.sum()
        return imp / s if s > 0 else imp


def oob_scorer(estimator, X, y):
    return estimator.oob_score_


def main():
    data = load_breast_cancer()
    X, y = data.data, data.target
    names = data.feature_names

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10],
        "max_features": ["sqrt", "log2"],
        "min_samples_split": [2, 5],
    }

    n = len(X_tr)
    dummy_cv = [(np.arange(n), np.arange(n))]

    gs = GridSearchCV(
        RandomForest(random_state=42),
        param_grid,
        scoring=oob_scorer,
        cv=dummy_cv,
        refit=True,
        verbose=1,
    )
    gs.fit(X_tr, y_tr)

    print("\nBest params:", gs.best_params_)
    print("Best OOB:", round(gs.best_score_, 4))

    rf = gs.best_estimator_

    t0 = time.perf_counter()
    rf.fit(X_tr, y_tr)
    t_custom = time.perf_counter() - t0

    acc = accuracy_score(y_te, rf.predict(X_te))
    print(f"\nCustom RF: acc={acc:.4f}, OOB={rf.oob_score_:.4f}, t={t_custom:.3f}s")

    imp = rf.oob_feature_importance(X_tr, y_tr)
    idx = np.argsort(imp)[::-1][:10]

    print("\nTop features:")
    for i in idx:
        print(f"{names[i]:30s} {imp[i]:.4f}")

    t0 = time.perf_counter()
    sk = SklearnRF(oob_score=True, random_state=42, **gs.best_params_)
    sk.fit(X_tr, y_tr)
    t_sk = time.perf_counter() - t0

    acc_sk = accuracy_score(y_te, sk.predict(X_te))
    print(f"\nsklearn RF: acc={acc_sk:.4f}, OOB={sk.oob_score_:.4f}, t={t_sk:.3f}s")


if __name__ == "__main__":
    main()