
import mlflow
from mlflow import MlflowClient

# Set the experiment name
mlflow.set_experiment("Vehicle_Damage_Classification")

# Results from our completed experiments
experiments = [
    {
        "name": "Baseline_MobileNetV2",
        "epochs": 3,
        "learning_rate": 0.001,
        "accuracy": 0.6667,
        "moderate_precision": 0.53,
        "moderate_recall": 0.41,
        "moderate_f1": 0.47,
        "macro_f1": 0.65
    },
    {
        "name": "Augmented_MobileNetV2",
        "epochs": 10,
        "learning_rate": 0.001,
        "accuracy": 0.6992,
        "moderate_precision": 0.53,
        "moderate_recall": 0.52,
        "moderate_f1": 0.53,
        "macro_f1": 0.69
    },
    {
        "name": "FineTuned_MobileNetV2",
        "epochs": 5,
        "learning_rate": 0.00001,
        "accuracy": 0.7236,
        "moderate_precision": 0.57,
        "moderate_recall": 0.59,
        "moderate_f1": 0.58,
        "macro_f1": 0.72
    }
]

for experiment in experiments:

    with mlflow.start_run(run_name=experiment["name"]):

        # Log parameters
        mlflow.log_param("model", "MobileNetV2")
        mlflow.log_param("epochs", experiment["epochs"])
        mlflow.log_param("learning_rate", experiment["learning_rate"])

        # Log evaluation metrics
        for metric in [
            "accuracy",
            "moderate_precision",
            "moderate_recall",
            "moderate_f1",
            "macro_f1"
        ]:
            mlflow.log_metric(metric, experiment[metric])

        print("Logged experiment:", experiment["name"])

print("\nAll experiments recorded in MLflow!")
