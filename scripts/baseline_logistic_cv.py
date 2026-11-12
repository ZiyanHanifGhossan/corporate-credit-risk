import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

train_data = pd.read_csv("data/processed/train.csv")

X = train_data.drop(columns="default_next_year")
y = train_data["default_next_year"]

baseline_model = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ("scaler", RobustScaler()),
           ("logistic_regression", LogisticRegression(
        solver="newton-cholesky",
        C=0.1,
        max_iter=1000,
    )),
])

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

scores = cross_validate(
    baseline_model,
    X,
    y,
    cv=cv,
    scoring={
        "average_precision": "average_precision",
        "roc_auc": "roc_auc",
    },
)

print("Average Precision per fold:", scores["test_average_precision"].round(3))
print("ROC-AUC per fold:", scores["test_roc_auc"].round(3))
print(
    "Rata-rata Average Precision:",
    round(scores["test_average_precision"].mean(), 3),
)
print(
    "Patokan acak dari proporsi event:",
    round(y.mean(), 3),
)