from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# --------------------------------------------------------------------
# Project Paths
# --------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "NoShowAppointments.csv"

BRONZE_PATH = BASE_DIR / "artifacts" / "bronze_appointments"
SILVER_PATH = BASE_DIR / "artifacts" / "silver_appointments"
GOLD_PATH = BASE_DIR / "artifacts" / "gold_appointments"


# --------------------------------------------------------------------
# Spark Session
# --------------------------------------------------------------------

def create_spark_session():

    spark = (
        SparkSession.builder
        .appName("HealthcareAIPlatform-BatchETL")
        .master("local[*]")
        .getOrCreate()
    )

    return spark


# --------------------------------------------------------------------
# Bronze
# --------------------------------------------------------------------

def run_bronze(spark):

    raw_df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(str(DATA_PATH))
    )

    raw_df.write.mode("overwrite").parquet(str(BRONZE_PATH))

    return raw_df.count()


# --------------------------------------------------------------------
# Silver
# --------------------------------------------------------------------

def run_silver(spark):

    bronze_df = spark.read.parquet(str(BRONZE_PATH))

    silver_df = (

        bronze_df

        .dropDuplicates()

        .withColumnRenamed("No-show", "no_show")

        .withColumn(
            "no_show",
            F.when(F.col("no_show") == "Yes", 1).otherwise(0)
        )

        .withColumn(
            "Gender",
            F.when(F.col("Gender") == "F", "Female").otherwise("Male")
        )

        .filter(F.col("Age") >= 0)

        .withColumn(
            "ScheduledDay",
            F.to_timestamp("ScheduledDay")
        )

        .withColumn(
            "AppointmentDay",
            F.to_timestamp("AppointmentDay")
        )

    )

    silver_df.write.mode("overwrite").parquet(str(SILVER_PATH))

    return silver_df.count()


# --------------------------------------------------------------------
# Gold
# --------------------------------------------------------------------

def run_gold(spark):

    silver_df = spark.read.parquet(str(SILVER_PATH))

    gold_df = (

        silver_df

        .withColumn(

            "days_until_appointment",

            F.datediff(

                F.to_date("AppointmentDay"),

                F.to_date("ScheduledDay")

            )

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
            "Neighbourhood",
            "Scholarship",
            "Hipertension",
            "Diabetes",
            "Alcoholism",
            "Handcap",
            "sms_received_flag",
            "days_until_appointment",
            "chronic_disease_count",
            "no_show"

        )

        .filter(

            F.col("days_until_appointment") >= 0

        )

    )

    gold_df.write.mode("overwrite").parquet(str(GOLD_PATH))

    return gold_df.count()


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------

if __name__ == "__main__":

    spark = create_spark_session()

    bronze_count = run_bronze(spark)

    silver_count = run_silver(spark)

    gold_count = run_gold(spark)

    print("\n===============================")
    print("ETL COMPLETED SUCCESSFULLY")
    print("===============================")

    print(f"Bronze rows : {bronze_count}")
    print(f"Silver rows : {silver_count}")
    print(f"Gold rows   : {gold_count}")

    spark.stop()