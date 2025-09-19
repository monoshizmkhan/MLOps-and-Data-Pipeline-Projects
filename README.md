# MLOps and Data Pipeline Projects

This is a collection of toy projects to learn and showcase usage of several tools including Airflow, MLflow, PySpark, Docker.

The dataset used is a Boston housing dataset. This project mainly consists of two separate experiments. The first experiment consists of two tasks in a DAG. The first task loads, processes and splits the data. The second task trains a simple tensorflow model on the data as a regression task and evaluates the performance. Also using MLflow to keep track of hyperparameters and results. Run Boston_DAG.py to run this experiment.

The second experiment uses the same dataset to build a pipeline from loading the data, processing and splitting it to fitting a basic Linear Regression model on this data, making predictions and evaluating the model (MAE). Run PySpark_Exp.py to run this experiment.

You can also build and run the Dockerfile to run the PySpark experiment without having to install/load packages manually. To build the image, run -

```
docker build -t <any_name_you_wish> .
```

Then to run the PySpark experiment, run -

```
docker run --rm <chosen_name>
```
