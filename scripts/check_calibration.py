from pathlib import Path

import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss


PROJECT_DIR = Path(__file__).resolve().parents[1]

# Membaca label sebenarnya dan probabilitas dari evaluasi holdout
data = pd.read_csv(
    PROJECT_DIR
    / "data"
    / "processed"
    / "holdout_predictions.csv"
)

actual = data["actual_bankruptcy"]

score_columns = {
    "Logistic Regression": "logistic_probability",
    "HistGradientBoosting": "hist_gradient_boosting_probability",
}

print(f"Proporsi bangkrut aktual: {actual.mean():.3f}")

for model_name, score_column in score_columns.items():
    probabilities = data[score_column]

    # Brier dan Log Loss mengukur kualitas probabilitas; nilai lebih rendah lebih baik
    brier = brier_score_loss(actual, probabilities)
    loss = log_loss(actual, probabilities)

    print(f"\n=== {model_name} ===")
    print(f"Brier score: {brier:.4f}")
    print(f"Log loss: {loss:.4f}")

    # Membagi perusahaan menjadi 10 kelompok berdasarkan skor dari rendah ke tinggi
    table = pd.DataFrame({
        "actual": actual,
        "predicted_probability": probabilities,
    })
    table["risk_group"] = pd.qcut(
        table["predicted_probability"],
        q=10,
        duplicates="drop",
    )

    # Membandingkan rata-rata probabilitas dengan tingkat bangkrut aktual per kelompok
    calibration_table = (
        table.groupby("risk_group", observed=True)
        .agg(
            jumlah_perusahaan=("actual", "size"),
            jumlah_bangkrut=("actual", "sum"),
            rata_rata_probabilitas=("predicted_probability", "mean"),
            tingkat_bangkrut_aktual=("actual", "mean"),
        )
    )

    print(calibration_table.to_string(
        float_format=lambda value: f"{value:.3f}"
    ))