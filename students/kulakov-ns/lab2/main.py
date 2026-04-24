import json
import time

from models.my_forest import get_my_forest
from models.sklearn_forest import get_sklearn_forest
from utils.dataset import load_titanic
from utils.metrics import evaluate_model



def make_report(my_result, sklearn_result, my_time, sklearn_time, importance):
    with open("data/report_template.md", "r", encoding="utf-8") as file:
        content = file.read()

    importance_rows = "\n".join(
        [f"| {feature} | {value:.4f} |" for feature, value in importance.items()]
    )

    content = content.format(
        my_params=json.dumps(my_result["best_params_"], ensure_ascii=False),
        sk_params=json.dumps(sklearn_result["best_params_"], ensure_ascii=False),
        my_oob=my_result["best_estimator_"].oob_score_,
        sk_oob=sklearn_result["best_estimator_"].oob_score_,
        my_acc=my_result["metrics"]["accuracy"],
        my_prec=my_result["metrics"]["precision"],
        my_rec=my_result["metrics"]["recall"],
        my_f1=my_result["metrics"]["f1"],
        my_time=my_time,
        sk_acc=sklearn_result["metrics"]["accuracy"],
        sk_prec=sklearn_result["metrics"]["precision"],
        sk_rec=sklearn_result["metrics"]["recall"],
        sk_f1=sklearn_result["metrics"]["f1"],
        sk_time=sklearn_time,
        importance_rows=importance_rows,
    )

    with open("data/report.md", "w", encoding="utf-8") as file:
        file.write(content)



def main():
    X_train, X_test, y_train, y_test, _ = load_titanic()

    start = time.perf_counter()
    my_grid = get_my_forest(X_train, y_train)
    my_time = time.perf_counter() - start

    start = time.perf_counter()
    sklearn_grid = get_sklearn_forest(X_train, y_train)
    sklearn_time = time.perf_counter() - start

    my_metrics = evaluate_model(
        "My RandomForest",
        my_grid.best_estimator_,
        X_test,
        y_test,
    )
    sklearn_metrics = evaluate_model(
        "Sklearn RandomForestClassifier",
        sklearn_grid.best_estimator_,
        X_test,
        y_test,
    )

    my_result = {
        "best_params_": my_grid.best_params_,
        "best_estimator_": my_grid.best_estimator_,
        "metrics": my_metrics,
    }
    sklearn_result = {
        "best_params_": sklearn_grid.best_params_,
        "best_estimator_": sklearn_grid.best_estimator_,
        "metrics": sklearn_metrics,
    }

    importance = dict(
        sorted(
            my_grid.best_estimator_.feature_importances_oob_.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )

    make_report(
        my_result=my_result,
        sklearn_result=sklearn_result,
        my_time=my_time,
        sklearn_time=sklearn_time,
        importance=importance,
    )

    with open("data/results.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "my_best_params": my_grid.best_params_,
                "my_oob": my_grid.best_estimator_.oob_score_,
                "my_metrics": my_metrics,
                "my_train_time": my_time,
                "my_feature_importances_oob": importance,
                "sklearn_best_params": sklearn_grid.best_params_,
                "sklearn_oob": sklearn_grid.best_estimator_.oob_score_,
                "sklearn_metrics": sklearn_metrics,
                "sklearn_train_time": sklearn_time,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
