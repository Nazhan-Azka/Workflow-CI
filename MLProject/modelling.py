from pathlib import Path
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def main():
    BASE_DIR = Path(__file__).resolve().parent

    # Dataset path (sesuaikan dengan folder kamu)
    data_path = BASE_DIR / "namadataset_prepocessing" / "listening_history_preprocessed.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {data_path}")

    df = pd.read_csv(data_path)

    X = df[["duration_sec", "hour"]]
    y = df["skipped"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ✅ Penting: gunakan run yang sudah dibuat oleh MLflow Project kalau ada
    # start_run() tanpa args akan "resume" active run dari MLflow Project
    with mlflow.start_run():
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred, zero_division=0))

        # log model sebagai artifact
        mlflow.sklearn.log_model(model, artifact_path="model")

    print("CI SUCCESS: training & logging selesai.")


if __name__ == "__main__":
    main()
