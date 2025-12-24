from pathlib import Path
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def main():
    BASE_DIR = Path(__file__).resolve().parent

    # ✅ CI friendly: tracking ke folder lokal (biar run+artifact tersimpan)
    mlflow.set_tracking_uri(f"file:{(BASE_DIR / 'mlruns').as_posix()}")
    mlflow.set_experiment("ci_experiment")

    # ✅ autolog (tanpa logging manual)
    mlflow.sklearn.autolog(log_models=True)

    # dataset path
    data_path = BASE_DIR / "namadataset_prepocessing" / "listening_history_preprocessed.csv"
    if not data_path.exists():
        alt_path = BASE_DIR / "namadataset_preprocessing" / "listening_history_preprocessed.csv"
        if alt_path.exists():
            data_path = alt_path
        else:
            raise FileNotFoundError(f"Dataset tidak ditemukan: {data_path} / {alt_path}")

    df = pd.read_csv(data_path)

    X = df[["duration_sec", "hour"]]
    y = df["skipped"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ✅ JANGAN pakai mlflow.start_run() di MLflow Project
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    print("DONE: MLflow Project training selesai (autolog menyimpan metrics & model).")


if __name__ == "__main__":
    main()
