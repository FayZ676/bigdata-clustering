from numpy import array
from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.clustering import KMeans
from pyspark import SparkContext
from pyspark.sql import SparkSession
import pyspark.sql.types as T
import matplotlib.pyplot as plt
import numpy as np
import time

spark = SparkSession.builder.master("local[*]") \
        .config('spark.jar.packages', 'gcs-connector-hadoop2-latest.jar') \
        .getOrCreate()

# Congfigure spark session
conf = spark.sparkContext._jsc.hadoopConfiguration()
conf.set("fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem")
spark.conf.set("fs.AbstractFileSystem.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS")

tick = time.time()

# Write dataset to spark dataframe
path = "gs://se513/newyorktlc2016/nytlc_*.csv"
data = spark.read.format("csv").option("header", "true").load(path)
data = data.na.drop()
data = data.select(col('trip_distance').cast('float'), col('tip_amount').cast('float'))
data = data.limit(10000)

# Vectorize seperated variables
cols = data.columns
assemble = VectorAssembler(inputCols=cols, outputCol = 'features')
assembled_data = assemble.transform(data)
data = assembled_data.select(col('features'))

# Compute best K using silhouette score
# sample = data.sample(False, 0.1, seed=42)
silhouette_score = []
total_sscore = 0
start = 2
end = 20
iterations = end - start
evaluator = ClusteringEvaluator(predictionCol='prediction', featuresCol = 'features', metricName = 'silhouette', distanceMeasure = 'squaredEuclidean')

for k in range(start, end):
        kmeans = KMeans(featuresCol = 'features', k = k)
        model = kmeans.fit(data)
        predictions = model.transform(data)
        score = evaluator.evaluate(predictions)
        silhouette_score.append(score) 

# Visualizing the silouette for best value of K
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(range(start, end), silhouette_score)
ax.set_xlabel('k')
ax.set_ylabel('tips')
fig.savefig('silhouette8888.jpg')

tock = time.time() - tick

print("Process complete")
print(tock)

