#-------------------------------------
# Libraries
#------------------------------------
import pyspark
from pyspark.sql.types import *
from pyspark.sql.functions import explode, col
from pyspark.ml.recommendation import ALS, ALSModel
import pandas as pd

#-------------------------------------
# Functions
#------------------------------------
def get_top_n_recs_for_user(spark, model, topn=10):
    """
    Given requested n and ALS model, returns top n recommended comics
    """
    tgt_acct_id = input()

    # Create spark df manually
    a_schema = StructType([StructField("account_id", LongType())])

    # Init lists
    tgt_list = []
    acct_list = []
    tgt_list.append(int(tgt_acct_id))
    acct_list.append(tgt_list)

    # Create one-row spark df
    tgt_accts = spark.createDataFrame(acct_list, schema=a_schema) 

    # Get recommendations for user
    userSubsetRecs = model.recommendForUserSubset(tgt_accts, topn)
    userSubsetRecs.persist()

    # Flatten the recs list
    top_n_exploded = (userSubsetRecs.withColumn('tmp',explode('recommendations'))
            .select('account_id', col("tmp.comic_id"), col("tmp.rating")))
    top_n_exploded.persist()

    # Get comics titles
    comics = spark.read.json('raw_data/comics.json')
    comics.persist()
    
    # shorten with alias
    top_n = top_n_exploded.alias('topn')
    com = comics.alias('com')

    # Clean up the spark df to list of titles
    top_n_titles = (top_n.join(com.select('comic_id','comic_title')
                          ,top_n.comic_id==com.comic_id)
                 .select('comic_title'))
    top_n_titles.persist()

    # Cast to pandas df and return it
    top_n_df = top_n_titles.select('*').toPandas()
    top_n_df.index += 1
    
    return top_n_df

def get_top_n_new_recs(spark, model, topn=10):
    """
    Given requested n and ALS model, returns top n recommended comics
    """
    
    # Multiplicative buffer
    # Get n x topn, because we will screen out previously bought
    buffer = 3
    
    # Get account number from user
    tgt_acct_id = input()

    # To 'save' the account number, will put it into a spark dataframe
    # Create spark df manually
    a_schema = StructType([StructField("account_id", LongType())])

    # Init lists
    tgt_list = []
    acct_list = []
    tgt_list.append(int(tgt_acct_id))
    acct_list.append(tgt_list)

    # Create one-row spark df
    tgt_accts = spark.createDataFrame(acct_list, schema=a_schema) 
    
    # Get recommendations for user
    userSubsetRecs = model.recommendForUserSubset(tgt_accts, (topn*buffer))
    userSubsetRecs.persist()

    # Flatten the recs list
    top_n_exploded = (userSubsetRecs.withColumn('tmp',explode('recommendations'))
            .select('account_id', col("tmp.comic_id"), col("tmp.rating")))
    top_n_exploded.persist()

    # Get comics titles
    comics = spark.read.json('raw_data/comics.json')
    comics.persist()
    
    # Get account-comics summary (already bought)
    acct_comics = spark.read.json('support_data/acct_comics.json')
    acct_comics = (
                    acct_comics.withColumnRenamed('account_id','acct_id')
                    .withColumnRenamed('comic_id', 'cmc_id')
                  )
    acct_comics.persist()

    # shorten with alias
    top_n = top_n_exploded.alias('topn')
    com = comics.alias('com')
    ac = acct_comics.alias('ac')
    

    # Clean up the spark df to list of titles, and only include these
    # that are NOT on bought list
    top_n_titles = (
                    top_n.join(com.select('comic_id','comic_title')
                              ,top_n.comic_id==com.comic_id, "left")
                         .join(ac, [top_n.account_id==ac.acct_id,
                                        top_n.comic_id==ac.cmc_id], 'left')
                         .filter('ac.acct_id is null') 
                         .select('comic_title')
                   )
    top_n_titles.persist()

    # Cast to pandas df and return it
    top_n_df = top_n_titles.select('*').toPandas()
    top_n_df = top_n_df.head(topn)
    top_n_df.index += 1
    
    return top_n_df
