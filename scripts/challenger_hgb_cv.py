import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

train_data = pd.read_csv("data/processed/train.csv")

X = train_data.drop(columns="default_next_year")
y = train_data["default_next_year"]

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

model = HistGradientBoostingClassifier(
    random_state=42,
    early_stopping=False,
)

param_grid = {
    "learning_rate": [0.05, 0.1],
    "max_iter": [100, 200],
    "max_leaf_nodes": [15, 31],
    "l2_regularization": [0.0, 1.0],
}

search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring={
        "average_precision": "average_precision",
        "roc_auc": "roc_auc",
    },
    refit="average_precision",
    cv=cv,
    n_jobs=1,
    verbose=1,
)

search.fit(X, y)

results = pd.DataFrame(search.cv_results_)[[
    "param_learning_rate",
    "param_max_iter",
    "param_max_leaf_nodes",
    "param_l2_regularization",
    "mean_test_average_precision",
    "std_test_average_precision",
    "mean_test_roc_auc",
    "rank_test_average_precision",
]]

print("\nLima konfigurasi teratas:")
print(
    results.sort_values("rank_test_average_precision")
    .head(5)
    .to_string(index=False)
)

print("\nParameter terbaik:", search.best_params_)
print("Average Precision terbaik:", round(search.best_score_, 3))