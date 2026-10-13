import os

# Switches off an MLflow start-up message about tracing, which is not used
# here. It has to be set before MLflow is imported.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
