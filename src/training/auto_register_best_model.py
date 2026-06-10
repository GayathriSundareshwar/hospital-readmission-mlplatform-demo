### learning experiments to select the best model out of all based on a metric
import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "no_show_prediction"
MODEL_NAME = "NoShowPredictionModel"
METRIC_NAME = "roc_auc"

client = MlflowClient()

experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    raise ValueError(f"Experiment not found: {EXPERIMENT_NAME}")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=[f"metrics.{METRIC_NAME} DESC"],
    max_results=1,
)

if not runs:
    raise ValueError("No runs found.")

best_run = runs[0]
best_run_id = best_run.info.run_id
best_metric = best_run.data.metrics.get(METRIC_NAME)

print("Best run ID:", best_run_id)
print(f"Best {METRIC_NAME}:", best_metric)

model_uri = f"runs:/{best_run_id}/model"

registered_model = mlflow.register_model(
    model_uri=model_uri,
    name=MODEL_NAME,
)

print("Registered model name:", registered_model.name)
print("Registered model version:", registered_model.version)