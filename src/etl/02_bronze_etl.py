from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("HealthcareAIPlatform-Bronze")
    .master("local[*]")
    .getOrCreate()
)

raw_df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv("../data/NoShowAppointments.csv")
)


raw_df.write.mode("overwrite").parquet("../artifacts/bronze_appointments")

bronze_df = spark.read.parquet("../artifacts/bronze_appointments")

print("Bronze rows:", bronze_df.count())
bronze_df.printSchema()

spark.stop()    