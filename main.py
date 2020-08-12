from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import json
import operator,string

print("Session Started")
session = SparkSession.builder.appName('SparkSQLDataset').master("local[*]").getOrCreate()

file="in_final/title.ratings.tsv"
df=session.read.option("delimiter","\t").csv(file,header=True)
print("Reading {}".format(file))

df_casted=df.withColumn("averageRating",df["averageRating"].cast(FloatType()))

mean=df_casted.agg({"averageRating": "mean"}).collect()[0][0]
print("Mean:  {}".format(mean))

id_threshold=df_casted.filter(df_casted["averageRating"] > mean).filter(df_casted["numVotes"] > 1000).select("tconst")
print("Selected Item with rate greater than {} and with number of Votes greater than 1000".format(mean))

file2="in_final/IMDB_reviews.json"
df_reviews=session.read.json(file2).select("movie_id","review_text")
print("Reading {}".format(file2))


new_df=df_reviews.join(id_threshold,df_reviews["movie_id"]==id_threshold["tconst"],how="left").select("review_text")
print("Joined Two Dataframe")
RDD=new_df.rdd.map(list)


n=RDD.map(lambda x:str(x).translate(str.maketrans('', '', string.punctuation)).lower())
print("Removed punctuation and cast to lowercase")

words=n.flatMap(lambda line: line.split())
RDDstopWords=session.sparkContext.textFile("in_final/stopwords.txt")
RDDnoStopWords=words.subtract(RDDstopWords)
print("Removed StopWords")

wordCounts = RDDnoStopWords.countByValue()


sortedwordCounts=sorted(wordCounts.items(),key=operator.itemgetter(1),reverse=True)
print("Sorted words")
with open("record.txt","w") as f:
    print("Writing in txt....")
    for x in sortedwordCounts:
        f.writelines("{}: {} \n".format(x[0],x[1]))
        print("       {}: {}".format(x[0],x[1]))
session.stop()