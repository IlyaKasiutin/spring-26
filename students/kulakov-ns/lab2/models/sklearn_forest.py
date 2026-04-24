import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from models.my_forest import oob_scorer



def get_sklearn_forest(X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
    param_grid = {
        "n_estimators": [25, 50, 100],
        "max_depth": [4, 6, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", 0.5],
    }

    train_idx = np.arange(len(X_train))

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(
            criterion="gini",
            bootstrap=True,
            oob_score=True,
            random_state=42,
            n_jobs=1,
        ),
        param_grid=param_grid,
        scoring=oob_scorer,
        cv=[(train_idx, train_idx)],
        refit=True,
        n_jobs=1,
    )

    return grid_search.fit(X_train, y_train)
