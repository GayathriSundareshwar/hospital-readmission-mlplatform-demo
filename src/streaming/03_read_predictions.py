from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


BASE_DIR = Path(__file__).resolve().parents[2]

STREAM_OUTPUT_DIR = BASE_DIR / "data" / "stream_output"
FINAL_OUTPUT_PATH = BASE_DIR / "artifacts" / "streaming_predictions_final"


def create_spark_session():
    return (
        SparkSession.builder
        .appName("HealthcareAIPlatform-ReadStreamingPredictions")
        .master("local[*]")
        .getOrCreate()
    )


if __name__ == "__main__":
    spark = create_spark_session()

    predictions_df = spark.read.parquet(str(STREAM_OUTPUT_DIR / "*"))

    final_df = (
        predictions_df
        .withColumn(
            "prediction_label",
            F.when(F.col("predicted_no_show") == 1, "Likely No-show")
             .otherwise("Likely Show")
        )
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
            "no_show_probability",
            "predicted_no_show",
            "prediction_label",
        )
    )

    final_df.write.mode("overwrite").parquet(str(FINAL_OUTPUT_PATH))

    print("Streaming predictions consolidated")
    print("Total rows:", final_df.count())

    print("\nPrediction distribution:")
    final_df.groupBy("prediction_label").count().show()

    print("\nSample predictions:")
    final_df.show(30, truncate=False)

    spark.stop()