from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


BASE_DIR = Path(__file__).resolve().parents[3]
PREDICTIONS_PATH = BASE_DIR / "artifacts" / "batch_predictions"


spark = (
    SparkSession.builder
    .appName("ReadBatchPredictions")
    .master("local[*]")
    .getOrCreate()
)

predictions_df = spark.read.parquet(str(PREDICTIONS_PATH))

print("Total prediction rows:", predictions_df.count())

print("\nPrediction counts:")
predictions_df.groupBy("predicted_no_show").count().show()

print("\nPrediction percentages:")
(
    predictions_df
    .groupBy("predicted_no_show")
    .count()
    .withColumn("percentage", F.round(F.col("count") / predictions_df.count() * 100, 2))
    .show()
)

print("\nSample predictions:")
predictions_df.show(20, truncate=False)

spark.stop()