# Libraries
import numpy as np
from pyspark.sql.types import StructType, StructField, LongType
from pyspark.sql.functions import explode, isnan, col, lower
from pyspark.sql import DataFrame
from pyspark.ml.recommendation import ALS
import itertools
import time
from functools import reduce

# Functions
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
    top_n_exploded = (userSubsetRecs.withColumn('tmp',
                                                explode('recommendations'))
                      .select('account_id', col("tmp.comic_id"),
                              col("tmp.rating"))
                      )
    top_n_exploded.persist()

    # Get comics titles
    comics = spark.read.json('raw_data/comics.json')
    comics.persist()

    # shorten with alias
    top_n = top_n_exploded.alias('topn')
    com = comics.alias('com')

    # Clean up the spark df to list of titles
    top_n_titles = (top_n.join(com.select('comic_id', 'comic_title'),
                    top_n.comic_id == com.comic_id).select('comic_title'))
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
    top_n_exploded = (userSubsetRecs.withColumn('tmp',
                      explode('recommendations'))
                      .select('account_id',
                      col("tmp.comic_id"), col("tmp.rating")))
    top_n_exploded.persist()

    # Get comics titles
    comics = spark.read.json('raw_data/comics.json')
    comics.persist()

    # Get account-comics summary (already bought)
    acct_comics = spark.read.json('support_data/acct_comics.json')
    acct_comics = (
                    acct_comics.withColumnRenamed('account_id', 'acct_id')
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
                    top_n.join(com.select('comic_id', 'comic_title'),
                               top_n.comic_id == com.comic_id, "left")
                         .join(ac, [top_n.account_id == ac.acct_id,
                                    top_n.comic_id == ac.cmc_id], 'left')
                         .filter('ac.acct_id is null')
                         .select('comic_title')
                   )
    top_n_titles.persist()

    # Cast to pandas df and return it
    top_n_df = top_n_titles.select('*').toPandas()
    top_n_df = top_n_df.head(topn)
    top_n_df.index += 1

    print('Total Runtime: {:.2f} seconds'.format(time.time() - start_time))

    return top_n_df


def train_ALS(train, test, evaluator, num_iters, reg_params, ranks, alphas):
    """
    Grid Search Function to select the best model based on RMSE
    of hold-out data
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
    # best_rank = -1
    # best_regularization = 0
    # best_alpha = 1
    best_model = None

    # tuple up the lists
    combos = [num_iters, reg_params, ranks, alphas]
    combos_tup = list(itertools.product(*combos))

    # Init list for list of combos
    params_errs = []

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
                  coldStartStrategy='drop',  # Just for CV
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

        # Save best model to date
        if error < min_error:
            # best_rank = rank
            # best_regularization = reg
            # best_alpha = alpha
            best_model = model

        # Add error to tuple, append to list of param and their errors
        tup_list = list(tup)
        _ = tup_list.append(error)
        params_errs.append(tup_list)

    return best_model, params_errs


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
    for i in range(1, k):
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
                                                  [test, 1-test],
                                                  random_seed
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
# New User Support


def get_comic_ids_for_user(comics_df, read_comics_list):
    """
    Given spark DF of existing comics and list of comics to 'match'
    Return list of like comics from the DF
    """
    # Initialize
    similar_comics_list = []
    for comic in read_comics_list:
        # print(comic)
        # Search for comic in df
        matched_comics = (comics_df.filter(lower(comics_df['comic_title'])
                                           .contains(str.lower(comic)))
                          .select('comic_id').rdd
                          .flatMap(lambda x: x).collect()
                          )
        similar_comics_list.extend(matched_comics)

    return similar_comics_list


def create_acct_id(model_data):
    """
    Given model data, create new account id that is just the max existing +1
    """
    # Get max account id
    max_acct_id = model_data.agg({'account_id': 'max'}).collect()[0][0]

    # New Account id
    new_acct_id = max_acct_id + 1

    return new_acct_id


def add_new_user(model_data, new_comic_ids, new_acct_id, spark_instance):
    """
    Given existing model data and the comic ids for new user,
    add rows for the new user to model data
    """
#     # Get max account id
#     max_acct_id = model_data.agg({'account_id':'max'}).collect()[0][0]

    # Create spark Df of new rows
    new_rows = spark_instance.createDataFrame([
                (new_acct_id, 1, comic_id) for comic_id in new_comic_ids])

    # Append to existing model data
    model_data_new = model_data.union(new_rows)

    return model_data_new


def train_als(model_data, current_params):
    """
    Given training data and set of parameters
    Returns trained ALS model
    """
    # Create ALS instance for cv with our chosen parametrs
    als_train = ALS(maxIter=current_params.get('maxIter'),
                    rank=current_params.get('rank'),
                    userCol='account_id',
                    itemCol='comic_id',
                    ratingCol='bought',
                    implicitPrefs=True,
                    regParam=current_params.get('regParam'),
                    alpha=current_params.get('alpha'),
                    coldStartStrategy='nan',  # we want to drop so
                                              # can get through CV
                    seed=41916)

    model_train = als_train.fit(model_data)
    return model_train


def get_comics_to_rate(comics_df, training_comic_ids):
    """
    Given list of comic ids,
    returns list of ids from master list that don't match
    """
    new_comic_ids = (comics_df.select('comic_id').distinct()
                     .filter(~col('comic_id').isin(training_comic_ids))
                     .select('comic_id').rdd.flatMap(lambda x: x).collect()
                     )
    return new_comic_ids


def recommend_n_comics(top_n, new_comics_ids, account_id, als_model,
                       comics_df, spark_instance):
    """
    Given a list of new comics (to the user) and requested number N
    Return list of N comics, ordered descending by recommendation score
    """

    # Create spark Df of new rows
    comics_to_predict = (spark_instance.createDataFrame([
                        (account_id, 1, comic_id)
                         for comic_id in new_comics_ids])
                         .select(col('_1').alias('account_id'),
                         col('_2').alias('bought'),
                         col('_3').alias('comic_id'))
                         )

    # Get predictions
    test_preds = als_model.transform(comics_to_predict)
    test_preds.persist()

    # Alias
    cdf = comics_df.alias('cdf')
    tp = test_preds.alias('tp')

    # Query results
    results = (tp.join(cdf, tp.comic_id == cdf.comic_id)
               .filter(~isnan(col('prediction')))
               .orderBy('prediction', ascending=False)
               .select('comic_title', 'img_url')
               .limit(top_n)
               ).toPandas()

    return results


def make_comic_recommendations(reading_list, top_n, comics_df, train_data,
                               model_params, spark_instance):
    """
    Given a list of comic titles and request for N
    Return list of comics recommendations as a pandas dataframe
    """
    start_time = time.time()

    # Get best-matching comic IDs
    train_comic_ids = get_comic_ids_for_user(comics_df, reading_list)

    # Create new account number
    new_id = create_acct_id(train_data)

    # Add new account to training data
    train_data_new = add_new_user(train_data, train_comic_ids, new_id,
                                  spark_instance)
    train_data_new.persist()

    # Train new ALS model
    als_model = train_als(train_data_new, model_params)

    # Get list of comics to rate, exclude those already matched
    new_comics_ids = get_comics_to_rate(comics_df, train_comic_ids)

    # Get pandas df of top n recommended comics!
    top_n_comics_df = recommend_n_comics(top_n, new_comics_ids, new_id,
                                         als_model,
                                         comics_df,
                                         spark_instance
                                         )

    print('Total Runtime: {:.2f} seconds'.format(time.time() - start_time))
    return top_n_comics_df


# For loop will automatically create and store ALS models
def create_als_models_list(userCol, itemCol, ratingCol, ranks, max_iters
                           ,reg_params, alphas, seed=1234):
    """
    Create list of ALS models based on combinations from lists of parameter
    Returns:
    List of ALS models (spark ML)
    """
    # Intiialize model list
    model_list = []
    
    # Loop through params and create model for each param combination
    for r in ranks:
        for mi in max_iters:
            for rp in reg_params:
                for a in alphas:
                    model_list.append(ALS(userCol=userCol
                                          ,itemCol=itemCol
                                          ,ratingCol=ratingCol
                                          ,rank = r, maxIter = mi, regParam = rp
                                          ,alpha = a
                                          ,coldStartStrategy="drop"
                                          ,nonnegative=True
                                          ,implicitPrefs=True))
    return model_list


def calculate_ROEM(sql_context, predictions, user_col, rating_col):
    """
    Calculate Rank-Ordered Error Metric
    Return:
    ROEM metric (float)
    """
    #Creates predictions table that can be queried
    predictions.createOrReplaceTempView("predictions") 

    #Sum of total number of plays of all songs
    denominator = predictions.groupBy().sum(rating_col).collect()[0][0]

    #Calculating rankings of songs predictions by user
    sql_string = ( "SELECT {}, {}, ".format(user_col, rating_col) +
                  "PERCENT_RANK() OVER (PARTITION BY {} ORDER"
                  .format(user_col) 
                  + " BY prediction DESC) AS rank FROM predictions"
                 )

    sql_context.sql(sql_string).createOrReplaceTempView("rankings")

    #Multiplies the rank of each song by the number of plays for each user
    #and adds the products together
    sql_string_2 = "SELECT SUM({} * rank) FROM rankings".format(rating_col)
    
    numerator = sql_context.sql(sql_string_2).collect()[0][0]
    
    return numerator / denominator


def get_ROEMs(sql_context, model_list, train, test, user_col, rating_col):
    """
    Calculate Rank-Ordered Error Metric for a set of model
    Returns:
    List of ROEMs
    """
    # Init list for results
    ROEMS = []
    n = 1
    start_time = time.time()
    
    # Loop through model data; fit, predict, and calc metrics
    for model in model_list:
        fitted_model = model.fit(train)
        predictions = fitted_model.transform(test)
        ROEM = calculate_ROEM(sql_context, predictions, user_col, rating_col)
        
        ROEMS.append(ROEM)
        print ("Validation ROEM #{}: {}".format(n, ROEM))
        n+=1     
        
    print ('Total Runtime: {:.2f} seconds'.format(time.time() - start_time))
    
    return ROEMS


def get_best_model(errors_list, model_list):
    """
    Based on list of errors and list of models, return the model with 
    the minimum error
    Returns:
    model object
    """
    # Find the index of the smallest ROEM
    import numpy as np
    
    idx = np.argmin(errors_list)
    print("Index of smallest error:", idx)

    # Find ith element of ROEMS
    print("Smallest error: ", errors_list[idx])

    return model_list[idx]
        

def create_spark_5_fold_set(train, seed=1234):
    """
    Provided a spark dataframe return a list of 5 tuples, where each pair is a
    test and train set for cross validation (pyspark dataframes)
    """
    #Building 5 folds within the training set.
    train1, train2, train3, train4, train5 = (train.randomSplit(
                                              [0.2, 0.2, 0.2, 0.2, 0.2]
                                              ,seed=seed)
                                            )
    fold1 = train2.union(train3).union(train4).union(train5)
    fold2 = train3.union(train4).union(train5).union(train1)
    fold3 = train4.union(train5).union(train1).union(train2)
    fold4 = train5.union(train1).union(train2).union(train3)
    fold5 = train1.union(train2).union(train3).union(train4)
    
    # Create list of tuples of CV pairs
    foldlist = [(fold1, train1), (fold2, train2), (fold3, train3)
            , (fold4, train4), (fold5, train5)]

    return foldlist   

def perform_cv(folds, model, sql_context, user_col, rating_col):
    """
    """
    errors = []
    
    for ft_pair in folds:

        # Fits model to fold within training data
        fitted_model = model.fit(ft_pair[0])

        # Generates predictions using fitted_model on respective CV test data
        predictions = fitted_model.transform(ft_pair[1])

        # Generates and prints a ROEM metric CV test data
        error = calculate_ROEM(sql_context, predictions, user_col, rating_col)
        errors.append(error)
    
    return errors   