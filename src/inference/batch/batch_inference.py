from pathlib import Path

import mlflow.sklearn
import pandas as pd
from pyspark.sql import SparkSession


BASE_DIR = Path(__file__).resolve().parents[3]

GOLD_PATH = BASE_DIR / "artifacts" / "gold_appointments"
PREDICTIONS_PATH = BASE_DIR / "artifacts" / "batch_predictions"

MODEL_NAME = "NoShowPredictionModel"
MODEL_URI = f"models:/{MODEL_NAME}/latest"

THRESHOLD = 0.30


def create_spark_session():
    return (
        SparkSession.builder
        .appName("HealthcareAIPlatform-BatchInference")
        .master("local[*]")
        .getOrCreate()
    )


if __name__ == "__main__":
    spark = create_spark_session()

    gold_df = spark.read.parquet(str(GOLD_PATH))

    feature_df = (
        gold_df
        .select(
            "PatientId",
            "AppointmentID",
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
        )
        .dropna()
    )

    pdf = feature_df.toPandas()

    id_cols = pdf[["PatientId", "AppointmentID"]]
    model_input = pdf.drop(columns=["PatientId", "AppointmentID"])

    model = mlflow.sklearn.load_model(MODEL_URI)

    no_show_probabilities = model.predict_proba(model_input)[:, 1]
    predicted_no_show = (no_show_probabilities >= THRESHOLD).astype(int)

    results_pdf = pd.concat(
        [
            id_cols.reset_index(drop=True),
            model_input.reset_index(drop=True),
            pd.Series(no_show_probabilities, name="no_show_probability"),
            pd.Series(predicted_no_show, name="predicted_no_show"),
        ],
        axis=1,
    )

    predictions_df = spark.createDataFrame(results_pdf)

    predictions_df.write.mode("overwrite").parquet(str(PREDICTIONS_PATH))

    print("Batch inference completed")
    print("Prediction rows:", predictions_df.count())
    print(f"Threshold used: {THRESHOLD}")
    predictions_df.show(20, truncate=False)

    spark.stop()