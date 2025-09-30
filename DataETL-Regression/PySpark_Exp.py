from pyspark.sql import SparkSession
from pyspark.sql.functions import rand
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

def loadData(session, path):
  data = session.read.csv(path, header=True)
  return data

def splitData(data, weights):
  data = data.orderBy(rand())
  for column_name in data.columns:
      data = data.withColumn(column_name, col(column_name).cast("float"))

  train, test = data.randomSplit(weights, seed=None)
  return train, test

def scale(df, scaler_model=None, train=False):
  features = df.columns[:-1]
  output = df.columns[-1]

  assemble = VectorAssembler(inputCols=features, outputCol="features")
  df = assemble.transform(df)

  if train:
    scaler = StandardScaler(inputCol="features", outputCol="features_scaled")
    scaler_model = scaler.fit(df.select("features"))
  df = scaler_model.transform(df)
  return scaler_model, df

def trainModel(train):
  lr = LinearRegression(featuresCol="features_scaled", labelCol="MEDV", predictionCol="predictions", loss="squaredError", maxIter=300)
  model = lr.fit(train)
  return model

def getPredictions(model, test):
  predictions = model.transform(test)
  return predictions

def evaluate(predictions):
  evaluator = RegressionEvaluator(predictionCol="predictions", labelCol="MEDV")
  results = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
  return results

def pipeline():
  spark = SparkSession.builder.appName("Toy project").config("spark.memory.offHeap.enabled","true").config("spark.memory.offHeap.size","1g").getOrCreate()
  data = loadData(spark, "Boston.csv")
  train, test = splitData(data, [0.8, 0.2])
  scaler, train = scale(train, None, True)
  _, test = scale(test, scaler, False)
  model = trainModel(train)
  predictions = getPredictions(model, test)
  results = evaluate(predictions)
  print(results)

pipeline()