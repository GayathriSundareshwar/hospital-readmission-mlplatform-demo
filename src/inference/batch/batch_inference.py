from pathlib import Path

import mlflow.pyfunc
import pandas as pd
from pyspark.sql import SparkSession


BASE_DIR = Path(__file__).resolve().parents[3]
print("Base directory:", BASE_DIR)
GOLD_PATH = BASE_DIR / "artifacts" / "gold_appointments"
PREDICTIONS_PATH = BASE_DIR / "artifacts" / "batch_predictions"

MODEL_NAME = "NoShowPredictionModel"
MODEL_URI = f"models:/{MODEL_NAME}/latest"


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

    model = mlflow.pyfunc.load_model(MODEL_URI)

    predictions = model.predict(model_input)

    results_pdf = pd.concat(
        [
            id_cols.reset_index(drop=True),
            model_input.reset_index(drop=True),
            pd.Series(predictions, name="predicted_no_show"),
        ],
        axis=1,
    )

    predictions_df = spark.createDataFrame(results_pdf)

    predictions_df.write.mode("overwrite").parquet(str(PREDICTIONS_PATH))

    print("Batch inference completed")
    print("Prediction rows:", predictions_df.count())
    predictions_df.show(10, truncate=False)

    spark.stop()