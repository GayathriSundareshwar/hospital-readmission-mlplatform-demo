from pathlib import Path

import mlflow.sklearn
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    DoubleType,
    IntegerType,
    StringType,
)


BASE_DIR = Path(__file__).resolve().parents[2]

STREAM_INPUT_DIR = BASE_DIR / "data" / "stream_input"
STREAM_OUTPUT_DIR = BASE_DIR / "data" / "stream_output"
CHECKPOINT_DIR = BASE_DIR / "checkpoints" / "stream_v1"

MODEL_NAME = "NoShowPredictionModel"
MODEL_URI = f"models:/{MODEL_NAME}/latest"

THRESHOLD = 0.30


schema = StructType([
    StructField("PatientId", DoubleType(), True),
    StructField("AppointmentID", IntegerType(), True),
    StructField("Gender", StringType(), True),
    StructField("ScheduledDay", StringType(), True),
    StructField("AppointmentDay", StringType(), True),
    StructField("Age", IntegerType(), True),
    StructField("Neighbourhood", StringType(), True),
    StructField("Scholarship", IntegerType(), True),
    StructField("Hipertension", IntegerType(), True),
    StructField("Diabetes", IntegerType(), True),
    StructField("Alcoholism", IntegerType(), True),
    StructField("Handcap", IntegerType(), True),
    StructField("SMS_received", IntegerType(), True),
    StructField("No-show", StringType(), True),
])


def create_spark_session():
    return (
        SparkSession.builder
        .appName("HealthcareAIPlatform-StreamingInference")
        .master("local[*]")
        .getOrCreate()
    )


def predict_micro_batch(batch_df, batch_id):
    if batch_df.count() == 0:
        return

    print(f"\nProcessing micro-batch: {batch_id}")
    print("Rows:", batch_df.count())

    feature_df = (
        batch_df
        .withColumnRenamed("No-show", "no_show")
        .withColumn("Gender", F.when(F.col("Gender") == "F", "Female").otherwise("Male"))
        .withColumn("ScheduledDay", F.to_timestamp("ScheduledDay"))
        .withColumn("AppointmentDay", F.to_timestamp("AppointmentDay"))
        .withColumn(
            "days_until_appointment",
            F.datediff(F.to_date("AppointmentDay"), F.to_date("ScheduledDay"))
        )
        .withColumn(
            "age_group",
            F.when(F.col("Age") < 18, "child")
             .when(F.col("Age") < 40, "young_adult")
             .when(F.col("Age") < 65, "adult")
             .otherwise("senior")
        )
        .withColumn(
            "chronic_disease_count",
            F.col("Hipertension") + F.col("Diabetes")
        )
        .withColumn(
            "sms_received_flag",
            F.col("SMS_received").cast("int")
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
        )
        .filter(F.col("days_until_appointment") >= 0)
        .dropna()
    )

    pdf = feature_df.toPandas()

    if pdf.empty:
        print("No valid rows after feature engineering.")
        return

    id_cols = pdf[["PatientId", "AppointmentID"]]
    model_input = pdf.drop(columns=["PatientId", "AppointmentID"])

    model = mlflow.sklearn.load_model(MODEL_URI)

    no_show_probabilities = model.predict_proba(model_input)[:, 1]
    predicted_no_show = (no_show_probabilities >= THRESHOLD).astype(int)

    result_pdf = pd.concat(
        [
            id_cols.reset_index(drop=True),
            model_input.reset_index(drop=True),
            pd.Series(no_show_probabilities, name="no_show_probability"),
            pd.Series(predicted_no_show, name="predicted_no_show"),
        ],
        axis=1,
    )

    result_df = batch_df.sparkSession.createDataFrame(result_pdf)

    output_path = STREAM_OUTPUT_DIR / f"batch_{batch_id}"
    result_df.write.mode("overwrite").parquet(str(output_path))

    print(f"Wrote predictions to: {output_path}")
    result_df.show(truncate=False)


if __name__ == "__main__":
    spark = create_spark_session()

    stream_df = (
        spark.readStream
        .schema(schema)
        .json(str(STREAM_INPUT_DIR))
    )

    query = (
        stream_df.writeStream
        .foreachBatch(predict_micro_batch)
        .option("checkpointLocation", str(CHECKPOINT_DIR))
        .trigger(processingTime="5 seconds")
        .start()
    )

    print("Streaming inference started.")
    print(f"Watching: {STREAM_INPUT_DIR}")
    print("Press Ctrl+C to stop.")

    query.awaitTermination()