#-------------------------------------
# Libraries
#------------------------------------
import pyspark
from pyspark.sql.types import *
from pyspark.sql.functions import explode, col
from pyspark.ml.recommendation import ALS, ALSModel
import pandas as pd
import itertools
import time

#-------------------------------------
# Functions
#------------------------------------
def get_top_n_recs_for_user(spark, model, topn=10):
    """
    Given requested n and ALS model, returns top n recommended comics
    DEPRECATED FOR get_top_n_new_recs
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
    start_time = time.time()

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
    
    print ('Total Runtime: {:.2f} seconds'.format(time.time() - start_time))

    return top_n_df

def train_ALS(train, test, evaluator, num_iters, reg_params, ranks, alphas):
    """
    Grid Search Function to select the best model based on RMSE of hold-out data
    Inspired by      
    https://github.com/KevinLiao159/MyDataSciencePortfolio/blob/master
            /movie_recommender/movie_recommendation_using_ALS.ipynb
    """

    # initial
    min_error = float('inf')
    best_rank = -1
    best_regularization = 0
    best_alpha = 1
    best_model = None
    
    # tuple up the lists
    combos = [num_iters, reg_params, ranks, alphas]
    combos_tup = list(itertools.product(*combos))
    
    # Loop though combos
    for tup in combos_tup:
        num_iter = tup[0]
        reg = tup[1]
        rank = tup[2]
        alpha = tup[3]

        # train ALS model
        als = ALS(maxIter=num_iter,
              rank=rank,
              userCol='account_id',
              itemCol='comic_id',
              ratingCol='bought',
              implicitPrefs=True,
              regParam=reg,
              alpha=alpha,
              coldStartStrategy='drop', #Just for CV
              seed=41916)

        model = als.fit(train)

        # Generate predictions on Test
        predictions = model.transform(test)
        predictions.persist()

        error = evaluator.evaluate(predictions)

        print('{} iterations, '.format(num_iter) + 
              '{} latent factors, regularization='.format(rank) +
              '{}, and alpha @ {} : '.format(reg, alpha) +
              'validation error is {:.4f}'.format(error))

        if error < min_error:
            best_rank = rank
            best_regularization = reg
            best_alpha = alpha
            best_model = model

    return best_model

