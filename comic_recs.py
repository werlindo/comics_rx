#-------------------------------------
# Libraries
#------------------------------------
import pyspark
from pyspark.sql.types import *
from pyspark.sql.functions import explode, col
from pyspark.sql import DataFrame
from pyspark.ml.recommendation import ALS, ALSModel
import pandas as pd
import itertools
import time
from functools import reduce

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
    Parameters
    ----------
    spark : spark instance
    model : FITTED ALS model
    topn : integer for now many results to return
    
    Returns
    -------
    pandas dataframe of top n comic recommendations
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
    Parameters
    ----------
    train : pyspark dataframe with training data
    test : pyspark dataframe with test data
    num_iters: list of iterations to test
    reg_params: list of regularization parameters to test
    ranks: list of # of latent factors to test
    alphas: list of alphas to test
    
    Returns
    -------
    fitted alsModel object
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

def get_spark_k_folds(spark_df, k=5, random_seed=1):
    """Take a spark df and split it into a list of k folds
    Parameters
    ----------
    spark_df : spark dataframe with dataset to train/test
    k : number of folds
    random_seed : if you want to push custom seed through randomSplit
    Returns
    -------
    list of k spark dataframes    
    """
    
    # Initialize dict to hold the folds
    folds = {}

    # Make copy of input df
    df = spark_df
    df.persist()
    
    # Loop through k's
    for i in range(1,k):
        test = 1/(k-i+1)
        part_1_nm = 'fold_' + str(i)
        
        # Name the train if not on last loop
        if i != (k-1):
            part_2_nm = 'train_' + str(i)
        else:
            part_2_nm = 'fold_' + str(i+1)

        # Run the splits
        folds[part_1_nm], folds[part_2_nm] = (
                                              df.randomSplit(
                                                  [test, 1-test]
                                                  ,random_seed
                                                  )
                                              )  

        # replace df if not on last loop
        df = folds[part_2_nm] if i != (k-1) else df 

        # drop the train sets from folds
        for key in list(folds.keys()):
            if 'train' in key:
                folds.pop(key)
        
        folds_list = [fold for fold in folds.values()]
        
    return folds_list

def get_cv_errors(folds, als, evaluator):
    """
    Given dictionary of spark DF folds and an ALS object
    returns list of errors
    Parameters
    ----------
    folds = list of spark dataframes
    als = ALS instance
    evaluator = spark Evaluator instance, usually regression
    
    Returns
    -------
    list of each test fold's prediction error metric
    """
    errors = []
    
    for i in range(len(folds)):

        # Partition out train and test
        test_fold_df = folds[i]

        train_folds = list(set(folds) - set([test_fold_df]))
        train_fold_df = reduce(DataFrame.unionAll, train_folds)
     
        # fit on train
        model = als.fit(train_fold_df)
        
        # get predictions on test
        preds = model.transform(test_fold_df)
        
        # Evaluate test
        errors.append(evaluator.evaluate(preds))
        
        # done
    return errors
    
