import os
import tempfile

import mlflow

mlflow.set_tracking_uri("http://localhost:5000/")
print("mlflow version:", mlflow.__version__)
mlflow.set_experiment("Default")

with tempfile.TemporaryDirectory() as tmp_dir, mlflow.start_run() as run:
    print("tracking uri:", mlflow.get_tracking_uri())
    print("artifact uri:", mlflow.get_artifact_uri())

    fname = "sample.txt"
    tmp_path = os.path.join(tmp_dir, fname)

    with open(tmp_path, "w") as f:
        f.write("sample")

    mlflow.log_param("p", 0)
    mlflow.log_metric("m", 1)
    mlflow.log_artifact(tmp_path)
