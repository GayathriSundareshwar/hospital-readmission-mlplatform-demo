from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("HealthcareAIPlatform-Silver")
    .master("local[*]")
    .getOrCreate()
)

bronze_df = spark.read.parquet("../artifacts/bronze_appointments")

silver_df = (
    bronze_df
    .dropDuplicates()
    .withColumnRenamed("No-show", "no_show")
    .withColumn("no_show", F.when(F.col("no_show") == "Yes", 1).otherwise(0))
    .withColumn("Gender", F.when(F.col("Gender") == "F", "Female").otherwise("Male"))
    .filter(F.col("Age") >= 0)
    .withColumn("ScheduledDay", F.to_timestamp("ScheduledDay"))
    .withColumn("AppointmentDay", F.to_timestamp("AppointmentDay"))
)

silver_df.write.mode("overwrite").parquet("../artifacts/silver_appointments")

print("Silver rows:", silver_df.count())
silver_df.printSchema()
silver_df.show(10, truncate=False)

spark.stop()