from pathlib import Path
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def main():
    BASE_DIR = Path(__file__).resolve().parent

    # ✅ Tracking pakai SQLite (aman di Windows)
    tracking_db = BASE_DIR / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db.as_posix()}")

    # ✅ Simpan artifacts ke folder lokal
    artifacts_dir = (BASE_DIR / "mlruns_artifacts").resolve()
    artifacts_dir.mkdir(exist_ok=True)

    # Create / set experiment dengan artifact_location yang jelas
    exp_name = "basic_experiment"
    exp = mlflow.get_experiment_by_name(exp_name)
    if exp is None:
        exp_id = mlflow.create_experiment(
            exp_name,
            artifact_location=artifacts_dir.as_uri()  # ini valid karena folder lokal biasa
        )
    else:
        exp_id = exp.experiment_id

    mlflow.set_experiment(experiment_id=exp_id)

    # ✅ Load dataset (sesuaikan dengan folder kamu yang typo)
    data_path = BASE_DIR / "namadataset_prepocessing" / "listening_history_preprocessed.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {data_path}")

    df = pd.read_csv(data_path)

    # Feature & target
    X = df[["duration_sec", "hour"]]
    y = df["skipped"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run(run_name="baseline_logreg_SUCCESS"):
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred, zero_division=0))

        # ✅ Simpan model sebagai artifact
        mlflow.sklearn.log_model(model, artifact_path="model")

    print("SUCCESS: run selesai dan model tersimpan sebagai artifact.")


if __name__ == "__main__":
    main()
