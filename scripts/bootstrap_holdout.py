from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler


PROJECT_DIR = Path(__file__).resolve().parents[1]
TARGET = "default_next_year"

train_df = pd.read_csv(PROJECT_DIR / "data" / "processed" / "train.csv")
holdout_df = pd.read_csv(PROJECT_DIR / "data" / "processed" / "test_holdout.csv")

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]
X_holdout = holdout_df.drop(columns=[TARGET])
y_holdout = holdout_df[TARGET].to_numpy()

models = {
    "logistic": make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        RobustScaler(),
        LogisticRegression(
            solver="newton-cholesky",
            C=1.0,
            max_iter=1000,
        ),
    ),
    "hist_gradient_boosting": HistGradientBoostingClassifier(
        l2_regularization=0.0,
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        random_state=42,
    ),
}

# Latih tiap model dan simpan skor risiko untuk holdout
probabilities = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    probabilities[name] = model.predict_proba(X_holdout)[:, 1]

# Simpan label asli dan skor model agar hasil evaluasi dapat ditinjau kembali
prediction_table = pd.DataFrame({
    "actual_bankruptcy": y_holdout,
    "logistic_probability": probabilities["logistic"],
    "hist_gradient_boosting_probability": probabilities["hist_gradient_boosting"],
})

prediction_path = PROJECT_DIR / "data" / "processed" / "holdout_predictions.csv"
prediction_table.to_csv(prediction_path, index=False)
print(f"Skor holdout disimpan di: {prediction_path}")

# Siapkan pengambilan sampel ulang dengan pengembalian
rng = np.random.default_rng(42)
n_rows = len(y_holdout)
top_count = int(np.ceil(0.10 * n_rows))
bootstrap_count = 2000

results = {
    name: {"AP": [], "ROC-AUC": [], "top10_precision": [], "top10_recall": []}
    for name in models
}

for _ in range(bootstrap_count):
    # Ambil 1.170 baris secara acak; satu baris dapat terambil lebih dari sekali
    sampled_indices = rng.integers(0, n_rows, size=n_rows)
    y_sample = y_holdout[sampled_indices]

    # Lewati sampel jika hanya berisi satu kelas
    if len(np.unique(y_sample)) < 2:
        continue

    for name in models:
        score_sample = probabilities[name][sampled_indices]

        ap = average_precision_score(y_sample, score_sample)
        roc_auc = roc_auc_score(y_sample, score_sample)

        top_indices = np.argsort(score_sample)[-top_count:]
        captured_events = int(y_sample[top_indices].sum())

        precision_top10 = captured_events / top_count
        recall_top10 = captured_events / int(y_sample.sum())

        results[name]["AP"].append(ap)
        results[name]["ROC-AUC"].append(roc_auc)
        results[name]["top10_precision"].append(precision_top10)
        results[name]["top10_recall"].append(recall_top10)

# Tampilkan batas bawah dan batas atas rentang 95%
for name, metrics in results.items():
    print(f"\n{name}")

    for metric_name, values in metrics.items():
        lower, upper = np.quantile(values, [0.025, 0.975])
        print(f"{metric_name}: {lower:.3f} sampai {upper:.3f}")