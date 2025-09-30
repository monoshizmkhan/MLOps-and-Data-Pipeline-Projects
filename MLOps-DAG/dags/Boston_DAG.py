from airflow.decorators import dag, task
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(__file__))
from dataLoader import dataLoader
import train_test
import mlflow
import numpy as np

@dag(
    start_date=datetime(year=2025, month=8, day=22, hour=3, minute=55),
    schedule="@daily",
    catchup=True,
    max_active_runs=2
)
def Boston():
    @task()
    def load_data():
        dl = dataLoader()
        train_x, train_y, val_x, val_y, test_x, test_y = dl.pipeline()
        np.save("train_x.npy", train_x)
        np.save("train_y.npy", train_y)
        np.save("val_x.npy", val_x)
        np.save("val_y.npy", val_y)
        np.save("test_x.npy", test_x)
        np.save("test_y.npy", test_y)


    @task()
    def train_and_test():
        train_x = np.load("train_x.npy")
        train_y = np.load("train_y.npy")
        val_x = np.load("val_x.npy")
        val_y = np.load("val_y.npy")
        test_x = np.load("test_x.npy")
        test_y = np.load("test_y.npy")
        with mlflow.start_run():
            variables, results_metrics = train_test.pipeline(train_x, train_y, val_x, val_y, test_x, test_y)
            for key, item in variables.items():
                mlflow.log_param(key, item)
            for key, item in results_metrics.items():
                mlflow.log_metric(key, item)

    # Set dependencies using function calls
    load_data.override(do_xcom_push=False)()
    train_and_test()

# Allow the DAG to be run
Boston()
