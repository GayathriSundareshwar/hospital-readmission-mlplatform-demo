from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("HealthcareAIPlatform-Gold")
    .master("local[*]")
    .getOrCreate()
)

silver_df = spark.read.parquet("../artifacts/silver_appointments")

gold_df = (
    silver_df
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
    .filter(F.col("days_until_appointment") >= 0)
)

gold_df.write.mode("overwrite").parquet("../artifacts/gold_appointments")

print("Gold rows:", gold_df.count())
gold_df.printSchema()
gold_df.show(10, truncate=False)

spark.stop()