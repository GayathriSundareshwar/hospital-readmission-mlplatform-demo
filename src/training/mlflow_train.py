from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parents[2]
GOLD_PATH = BASE_DIR / "artifacts" / "gold_appointments"
MODEL_PATH = BASE_DIR / "models" / "no_show_model_mlflow.pkl"


def create_spark_session():
    return (
        SparkSession.builder
        .appName("HealthcareAIPlatform-MLflowTraining")
        .master("local[*]")
        .getOrCreate()
    )


if __name__ == "__main__":
    spark = create_spark_session()

    gold_df = spark.read.parquet(str(GOLD_PATH))

    ml_df = (
        gold_df
        .select(
            "Gender",
            "Age",
            "age_group",
            "Scholarship",
            "Hipertension",
            "Diabetes",
            "Alcoholism",
            "Handcap",
            "sms_received_flag",
            "days_until_appointment",
            "chronic_disease_count",
            "no_show",
        )
        .dropna()
    )

    pdf = ml_df.toPandas()

    X = pdf.drop(columns=["no_show"])
    y = pdf["no_show"]

    categorical_cols = ["Gender", "age_group"]
    numeric_cols = [
        "Age",
        "Scholarship",
        "Hipertension",
        "Diabetes",
        "Alcoholism",
        "Handcap",
        "sms_received_flag",
        "days_until_appointment",
        "chronic_disease_count",
    ]

    n_estimators = 100
    max_depth = 8

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        class_weight="balanced",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    mlflow.set_experiment("no_show_prediction")

    with mlflow.start_run(run_name="random_forest_baseline"):
        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)
        probs = pipeline.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, preds)
        roc_auc = roc_auc_score(y_test, probs)

        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("features", list(X.columns))

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc_auc", roc_auc)

        mlflow.sklearn.log_model(pipeline, "model")

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)

        print("Accuracy:", accuracy)
        print("ROC AUC:", roc_auc)
        print(classification_report(y_test, preds))
        print(f"Local model saved to: {MODEL_PATH}")

    spark.stop()