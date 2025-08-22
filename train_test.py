import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_absolute_error, median_absolute_error

def createModel(layer_1_size=32):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(layer_1_size))
    model.add(tf.keras.layers.Dense(1))
    return model

def trainModel(layer_1_size,
               optimizer,
               train_x, train_y,
               val_x, val_y):
    model = createModel(layer_1_size=layer_1_size)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanAbsoluteError(), metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    model.fit(x=train_x, y=train_y, validation_data=(val_x, val_y), epochs=50, verbose=1, callbacks=[early_stopping])

    return model

def testModel(model, test_x):
    pred_y = model.predict(test_x, verbose=0)
    pred_y = np.squeeze(pred_y)
    return pred_y

def evaluate(test_y, pred_y):
    metrics = {}
    metrics["MAE"] = mean_absolute_error(test_y, pred_y)
    metrics["MdAE"] = median_absolute_error(test_y, pred_y)
    return metrics

def pipeline(train_x, train_y, val_x, val_y, test_x, test_y):
    variables = {"layer_1_size": 32,
                 "optimizer": "SGD"}

    model = trainModel(variables["layer_1_size"], variables["optimizer"], train_x, train_y, val_x, val_y)

    pred_y = testModel(model, test_x)

    results_metrics = evaluate(test_y, pred_y)

    return variables, results_metrics

# from dataLoader import dataLoader
# import mlflow
#
# with mlflow.start_run():
#     dl = dataLoader()
#
#     data = dl.pipeline()
#     variables, results_metrics = pipeline(data)
#     for key, item in variables.items():
#         mlflow.log_param(key, item)
#     for key, item in results_metrics.items():
#         mlflow.log_metric(key, item)
#     print(results_metrics)
