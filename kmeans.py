from numpy import array
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.clustering import KMeans
from pyspark import SparkContext
from pyspark.sql import SparkSession, SQLContext
import pyspark.sql.types as T
import matplotlib.pyplot as plt
import time

# Set up connection to Google Storage Bucket
spark = SparkSession.builder.master("local[*]") \
        .config('spark.jar.packages', 'gcs-connector-hadoop2-latest.jar') \
        .getOrCreate()

# Congfigure spark session
conf = spark.sparkContext._jsc.hadoopConfiguration()
conf.set("fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem")
spark.conf.set("fs.AbstractFileSystem.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS")
sqlContext = SQLContext(spark)

# Start timer to keep track of execution
tick = time.time()

# Write dataset to spark dataframe
path = "gs://se513/newyorktlc2016/nytlc_000000000000.csv"
data = spark.read.format("csv").option("header", "true").load(path)

# Drop null values, select columns, and convert to float
data = data.na.drop()
df = data.select(col('trip_distance').cast('float'), col('tip_amount').cast('float'))

# Split dataset into 80% train 20% test set
train, test = df.randomSplit([0.8, 0.2], seed=42)

# Vectorize train data and create features column for clustering
cols = train.columns
assemble = VectorAssembler(inputCols=cols, outputCol = 'features')
df_features = assemble.transform(train).select('features')
df_features.show(5)

# Train K Means model
kmeans = KMeans(featuresCol = 'features', k = 7, seed = 1)
model = kmeans.fit(df_features)
centers = model.clusterCenters()

print("Cluster Centers: ")
for center in centers:
    print(center)

# Vectorize test data for clustering
cols = test.columns
assemble = VectorAssembler(inputCols=cols, outputCol = 'features')
df_features = assemble.transform(test).select('features')
df_features.show(5)

# Test K Means model
kmeans = KMeans(featuresCol = 'features', k = 7, seed = 1)
model = kmeans.fit(df_features)
centers = model.clusterCenters()

print("Cluster Centers: ")
for center in centers:
    print(center)

summary = model.summary
print("Cluster Sizes: ",summary.clusterSizes)

tock = time.time() - tick
print("Total Execution Time: ",tock)
print("Process complete")

