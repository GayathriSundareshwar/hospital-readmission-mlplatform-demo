from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parents[2]
GOLD_PATH = BASE_DIR / "artifacts" / "gold_appointments"
COMPARISON_DIR = BASE_DIR / "artifacts" / "model_comparison"

EXPERIMENT_NAME = "no_show_prediction"
MODEL_NAME = "NoShowPredictionModel"
SELECTION_METRIC = "roc_auc"


def create_spark_session():
    return (
        SparkSession.builder
        .appName("HealthcareAIPlatform-ModelComparison")
        .master("local[*]")
        .getOrCreate()
    )


def build_model(model_type, params):
    if model_type == "logistic_regression":
        return LogisticRegression(
            C=params["C"],
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )

    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            class_weight="balanced",
            random_state=42,
        )

    if model_type == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            random_state=42,
        )

    raise ValueError(f"Unsupported model type: {model_type}")


def get_experiments():
    return [
        {"run_name": "lr_c_0_01", "model_type": "logistic_regression", "params": {"C": 0.01}},
        {"run_name": "lr_c_0_1", "model_type": "logistic_regression", "params": {"C": 0.1}},
        {"run_name": "lr_c_1", "model_type": "logistic_regression", "params": {"C": 1}},
        {"run_name": "lr_c_10", "model_type": "logistic_regression", "params": {"C": 10}},
        {"run_name": "lr_c_100", "model_type": "logistic_regression", "params": {"C": 100}},

        {"run_name": "rf_50_depth_5", "model_type": "random_forest", "params": {"n_estimators": 50, "max_depth": 5}},
        {"run_name": "rf_100_depth_8", "model_type": "random_forest", "params": {"n_estimators": 100, "max_depth": 8}},
        {"run_name": "rf_200_depth_10", "model_type": "random_forest", "params": {"n_estimators": 200, "max_depth": 10}},
        {"run_name": "rf_300_depth_12", "model_type": "random_forest", "params": {"n_estimators": 300, "max_depth": 12}},
        {"run_name": "rf_500_depth_none", "model_type": "random_forest", "params": {"n_estimators": 500, "max_depth": None}},

        {"run_name": "gb_50_lr_0_1_depth_3", "model_type": "gradient_boosting", "params": {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 3}},
        {"run_name": "gb_100_lr_0_1_depth_3", "model_type": "gradient_boosting", "params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3}},
        {"run_name": "gb_200_lr_0_1_depth_3", "model_type": "gradient_boosting", "params": {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 3}},
        {"run_name": "gb_100_lr_0_05_depth_3", "model_type": "gradient_boosting", "params": {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 3}},
        {"run_name": "gb_100_lr_0_2_depth_3", "model_type": "gradient_boosting", "params": {"n_estimators": 100, "learning_rate": 0.2, "max_depth": 3}},
    ]


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

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    mlflow.set_experiment(EXPERIMENT_NAME)

    results = []

    for exp in get_experiments():
        run_name = exp["run_name"]
        model_type = exp["model_type"]
        params = exp["params"]

        with mlflow.start_run(run_name=run_name) as run:
            model = build_model(model_type, params)

            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", model),
                ]
            )

            pipeline.fit(X_train, y_train)

            preds = pipeline.predict(X_test)

            if hasattr(pipeline.named_steps["model"], "predict_proba"):
                probs = pipeline.predict_proba(X_test)[:, 1]
            else:
                probs = preds

            accuracy = accuracy_score(y_test, preds)
            roc_auc = roc_auc_score(y_test, probs)
            precision = precision_score(y_test, preds, zero_division=0)
            recall = recall_score(y_test, preds, zero_division=0)
            f1 = f1_score(y_test, preds, zero_division=0)

            mlflow.log_param("model_type", model_type)
            for key, value in params.items():
                mlflow.log_param(key, value)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            mlflow.sklearn.log_model(pipeline, "model")

            result = {
                "run_id": run.info.run_id,
                "run_name": run_name,
                "model_type": model_type,
                "accuracy": accuracy,
                "roc_auc": roc_auc,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                **params,
            }

            results.append(result)

            print(f"Completed: {run_name}")
            print(f"ROC AUC: {roc_auc}")
            print("-" * 50)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by=SELECTION_METRIC, ascending=False)

    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    comparison_csv_path = COMPARISON_DIR / "model_comparison.csv"
    comparison_chart_path = COMPARISON_DIR / "model_comparison.png"

    results_df.to_csv(comparison_csv_path, index=False)

    top_df = results_df.head(15).sort_values(by=SELECTION_METRIC, ascending=True)

    plt.figure(figsize=(12, 8))
    plt.barh(top_df["run_name"], top_df[SELECTION_METRIC])
    plt.xlabel(SELECTION_METRIC)
    plt.ylabel("Experiment Run")
    plt.title("Model Comparison by ROC AUC")
    plt.tight_layout()
    plt.savefig(comparison_chart_path)

    best = results_df.iloc[0]
    best_run_id = best["run_id"]
    best_score = best[SELECTION_METRIC]

    model_uri = f"runs:/{best_run_id}/model"

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
    )

    print("\n===============================")
    print("MODEL COMPARISON COMPLETE")
    print("===============================")
    print(f"Comparison CSV: {comparison_csv_path}")
    print(f"Comparison Chart: {comparison_chart_path}")
    print(f"Best run: {best['run_name']}")
    print(f"Best model type: {best['model_type']}")
    print(f"Best {SELECTION_METRIC}: {best_score}")
    print(f"Registered model: {registered_model.name}")
    print(f"Registered version: {registered_model.version}")

    spark.stop()