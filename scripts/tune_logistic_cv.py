import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

train_data = pd.read_csv("data/processed/train.csv")

X = train_data.drop(columns="default_next_year")
y = train_data["default_next_year"]

pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ("scaler", RobustScaler()),
    ("logistic_regression", LogisticRegression(
        solver="newton-cholesky",
        max_iter=1000,
    )),
])

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

search = GridSearchCV(
    estimator=pipeline,
    param_grid={
        "logistic_regression__C": [0.01, 0.1, 1.0],
    },
    scoring="average_precision",
    cv=cv,
    refit=True,
)

search.fit(X, y)

results = pd.DataFrame(search.cv_results_)[[
    "param_logistic_regression__C",
    "mean_test_score",
    "std_test_score",
    "rank_test_score",
]]

print(results.sort_values("rank_test_score").to_string(index=False))
print("\nC terbaik:", search.best_params_["logistic_regression__C"])
print("Average Precision terbaik:", round(search.best_score_, 3))