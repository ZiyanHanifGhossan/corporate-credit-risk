from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler


# Menentukan lokasi folder utama projek dari lokasi script ini
PROJECT_DIR = Path(__file__).resolve().parents[1]

# Membaca data training dan holdout yang sudah disimpan sebelumnya
train_df = pd.read_csv(PROJECT_DIR / "data" / "processed" / "train.csv")
holdout_df = pd.read_csv(PROJECT_DIR / "data" / "processed" / "test_holdout.csv")

# Menentukan kolom target yang ingin diprediksi
TARGET = "default_next_year"

# Memisahkan fitur (rasio keuangan) dan target dari data training
X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

# Memisahkan fitur dan target dari data holdout
X_holdout = holdout_df.drop(columns=[TARGET])
y_holdout = holdout_df[TARGET]

# Membuat kandidat Logistic Regression:
# nilai kosong diisi median, indikator missing ditambahkan, fitur diskalakan
logistic_model = make_pipeline(
    SimpleImputer(strategy="median", add_indicator=True),
    RobustScaler(),
    LogisticRegression(
        solver="newton-cholesky",
        C=1.0,
        max_iter=1000,
    ),
)

# Membuat kandidat HistGradientBoosting dengan konfigurasi terbaik dari pencarian CV
boosting_model = HistGradientBoostingClassifier(
    l2_regularization=0.0,
    learning_rate=0.05,
    max_iter=200,
    max_leaf_nodes=15,
    random_state=42,
)

# Fungsi ini melatih satu model dan menghitung kinerjanya pada holdout
def evaluate_model(name, model):
    model.fit(X_train, y_train)

    # Mengambil probabilitas bahwa setiap perusahaan akan bangkrut
    probabilities = model.predict_proba(X_holdout)[:, 1]

    # Memilih 10% perusahaan dengan skor risiko tertinggi untuk contoh kapasitas tinjauan
    top_count = int(np.ceil(0.10 * len(y_holdout)))
    top_indices = np.argsort(probabilities)[-top_count:]
    captured_events = int(y_holdout.iloc[top_indices].sum())

    # Menghitung metrik evaluasi
    ap = average_precision_score(y_holdout, probabilities)
    roc_auc = roc_auc_score(y_holdout, probabilities)
    precision_top_10 = captured_events / top_count
    recall_top_10 = captured_events / int(y_holdout.sum())

    print(f"\n{name}")
    print(f"Average Precision: {ap:.3f}")
    print(f"ROC-AUC: {roc_auc:.3f}")
    print(
        f"Di kelompok risiko tertinggi ({top_count} perusahaan): "
        f"{captured_events} dari {int(y_holdout.sum())} kasus bangkrut"
    )
    print(f"Persentase bangkrut di kelompok top 10%: {precision_top_10:.1%}")
    print(f"Bagian seluruh kasus bangkrut yang tertangkap: {recall_top_10:.1%}")


# Menampilkan jumlah kasus bangkrut dan patokan AP acak pada holdout
print(f"Jumlah baris holdout: {len(y_holdout)}")
print(f"Jumlah kasus bangkrut holdout: {int(y_holdout.sum())}")
print(f"Patokan Average Precision acak: {y_holdout.mean():.3f}")

# Menilai kedua kandidat pada baris holdout yang sama
evaluate_model("Logistic Regression", logistic_model)
evaluate_model("HistGradientBoosting", boosting_model)