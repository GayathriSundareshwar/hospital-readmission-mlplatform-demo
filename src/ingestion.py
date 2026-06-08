from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("HealthcareAIPlatform-Ingestion")
    .master("local[*]")
    .getOrCreate()
)

appointments_df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv("../data/NoShowAppointments.csv")
)

appointments_df.show(10)
print("Total rows:", appointments_df.count())
appointments_df.printSchema()

spark.stop()