import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

train_data = pd.read_csv("data/processed/train.csv")

y = train_data["default_next_year"]
X_missing = train_data.drop(columns="default_next_year").isna().astype(int)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

model = LogisticRegression(
    solver="liblinear",
    max_iter=1000,
)

scores = cross_validate(
    model,
    X_missing,
    y,
    cv=cv,
    scoring={
        "average_precision": "average_precision",
        "roc_auc": "roc_auc",
    },
)

print("Jumlah indikator missing:", X_missing.shape[1])
print("Average Precision per fold:", scores["test_average_precision"].round(3))
print("ROC-AUC per fold:", scores["test_roc_auc"].round(3))
print(
    "Rata-rata Average Precision:",
    round(scores["test_average_precision"].mean(), 3),
)