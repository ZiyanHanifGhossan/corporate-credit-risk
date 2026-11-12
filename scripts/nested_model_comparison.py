from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler


PROJECT_DIR = Path(__file__).resolve().parents[1]
TARGET = "default_next_year"

# Nested CV memakai seluruh data deduplicated, bukan holdout lama
data = pd.read_csv(
    PROJECT_DIR / "data" / "processed" / "polish_5year_deduplicated.csv"
)

X = data.drop(columns=[TARGET])
y = data[TARGET]

# Kandidat Logistic Regression beserta nilai C yang akan dicari
logistic_model = make_pipeline(
    SimpleImputer(strategy="median", add_indicator=True),
    RobustScaler(),
    LogisticRegression(
        solver="newton-cholesky",
        max_iter=1000,
    ),
)

logistic_grid = {
    "logisticregression__C": [0.1, 1.0, 10.0],
}

# Kandidat HistGradientBoosting dan konfigurasi yang akan dicari
boosting_model = HistGradientBoostingClassifier(
    random_state=42
)

boosting_grid = {
    "learning_rate": [0.025, 0.05],
    "max_iter": [200, 300],
    "max_leaf_nodes": [15, 31],
    "l2_regularization": [0.0, 1.0],
}

model_specs = {
    "Logistic Regression": (logistic_model, logistic_grid),
    "HistGradientBoosting": (boosting_model, boosting_grid),
}

# Outer CV menguji model pada bagian data yang tidak ikut dalam pencarian
outer_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

# Menyimpan metrik dari setiap fold
fold_results = []

for fold_number, (train_indices, validation_indices) in enumerate(
    outer_cv.split(X, y),
    start=1,
):
    print(f"\nMemulai fold luar {fold_number} dari 5")

    X_outer_train = X.iloc[train_indices]
    y_outer_train = y.iloc[train_indices]
    X_outer_validation = X.iloc[validation_indices]
    y_outer_validation = y.iloc[validation_indices]

    # Inner CV memilih konfigurasi terbaik hanya dari data outer training
    inner_cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=100 + fold_number,
    )

    for model_name, (estimator, parameter_grid) in model_specs.items():
        print(f"Mencari konfigurasi terbaik: {model_name}")

        search = GridSearchCV(
            estimator=estimator,
            param_grid=parameter_grid,
            scoring="average_precision",
            cv=inner_cv,
            refit=True,
            n_jobs=1,
        )

        search.fit(X_outer_train, y_outer_train)

        # Menilai model terpilih pada fold luar yang belum dipakai untuk pencarian
        probabilities = search.best_estimator_.predict_proba(
            X_outer_validation
        )[:, 1]

        ap = average_precision_score(
            y_outer_validation, probabilities
        )
        roc_auc = roc_auc_score(
            y_outer_validation, probabilities
        )

        # Menghitung kinerja bila tim hanya meninjau 10% dengan risiko tertinggi
        top_count = int(np.ceil(0.10 * len(y_outer_validation)))
        top_indices = np.argsort(probabilities)[-top_count:]
        captured_events = int(
            y_outer_validation.iloc[top_indices].sum()
        )

        precision_top10 = captured_events / top_count
        recall_top10 = captured_events / int(y_outer_validation.sum())

        fold_results.append({
            "model": model_name,
            "fold": fold_number,
            "average_precision": ap,
            "roc_auc": roc_auc,
            "precision_top10": precision_top10,
            "recall_top10": recall_top10,
        })

        print(f"AP fold ini: {ap:.3f}")
        print(f"ROC-AUC fold ini: {roc_auc:.3f}")
        print(f"Konfigurasi terpilih: {search.best_params_}")

# Menampilkan ringkasan rata-rata dan variasi hasil antar-fold
results_df = pd.DataFrame(fold_results)

for model_name in model_specs:
    model_results = results_df[results_df["model"] == model_name]

    print(f"\n=== {model_name} ===")

    for metric in [
        "average_precision",
        "roc_auc",
        "precision_top10",
        "recall_top10",
    ]:
        values = model_results[metric]

        print(
            f"{metric}: rata-rata {values.mean():.3f}, "
            f"standar deviasi {values.std():.3f}"
        )

# Menyimpan ringkasan agar dapat diperiksa atau digunakan dalam laporan
output_path = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "nested_cv_results.csv"
)
results_df.to_csv(output_path, index=False)
print(f"\nRingkasan hasil disimpan di: {output_path}")