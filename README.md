# MLOps and Data Pipeline Projects

This is a collection of toy projects to learn and showcase usage of several tools including Airflow, MLflow, PySpark, Docker. Kubernetes is the planned as the next tool to be showcased here using these projects.

The dataset used is a Boston housing dataset. This project mainly consists of two separate experiments.

The first experiment is in the MLOps-DAG folder and uses Airflow and PySpark. This exerpiment consists of two tasks in a DAG. The first task loads, processes and splits the data. The second task trains a simple tensorflow model on the data as a regression task and evaluates the performance. Also using MLflow to keep track of hyperparameters and results. To run this experiment manually, install the required packages, initialize Airflow, start the Airflow scheduler, and then run the Boston_DAG DAG (you can find the commands in the dockerfile).

The second experiment is in the DataETL-Regression folder and uses PySpark. This experiment uses the same Boston housing dataset to build a pipeline from loading the data, processing and splitting it to fitting a basic Linear Regression model on this data, making predictions and evaluating the model (MAE). Run PySpark_Exp.py to run this experiment.

You can also build and run the Dockerfile to run each experiment without having to install/load packages manually. To build the image, run -

```
docker build -t <any_name_you_wish> .
```

Then to run the experiment, run -

```
docker run --rm <chosen_name>
```
