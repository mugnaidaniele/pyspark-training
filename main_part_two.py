from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import json
import operator,string
from pyspark.sql.functions import udf


from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import ClusteringEvaluator




session = SparkSession.builder.appName('SparkSQLDataset').master("local[*]").getOrCreate()

df=session.read.option("delimiter","\t").csv("in_final/title.ratings.tsv",header=True)

df_filtered=df.filter(df["numvotes"]>0)

df_casted=df.withColumn("averageRating",df["averageRating"].cast(FloatType()))
"""
def filter_rate(rate):
    if rate <5.5:
        return 0
    elif rate>5.5:
        return 1
f=udf(filter_rate)
new_df=df_casted.withColumn("averageRating",f(df_casted.averageRating))
new_df.show()
"""

print("-------------------------------------------------------------------------------------")
kmeans=KMeans(k=8,seed=1)
a=VectorAssembler(inputCols=["averageRating"],outputCol="features")
trainingData=a.transform(df_casted)
model=kmeans.fit(trainingData.select("features"))

transformed=model.transform(trainingData)
#transformed.show()
evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(transformed)
print("Silhouette with squared euclidean distance = " + str(silhouette))


centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)

df_new=transformed.select("tconst","prediction")



