import random
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle
from flask import Flask, request, render_template, jsonify, session, redirect, make_response

# Pyspark imports
import pyspark
from pyspark.sql import SparkSession

import ast

from ..comic_recs import make_comic_recommendations

app = Flask(__name__, static_url_path="")

# spark config
spark = SparkSession \
    .builder \
    .appName("movie recommendation") \
    .config("spark.driver.maxResultSize", "1g") \
    .config("spark.driver.memory", "1g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.master", "local[*]") \
    .getOrCreate()

# spark = pyspark.sql.SparkSession.builder.master("local[*]").getOrCreate()

comics_df = spark.read.json('./comrx/dev/support_data/comics.json')
comics_df.persist()

comics_sold = spark.read.json('./comrx/dev/raw_data/als_input_filtered.json')
comics_sold.persist()

# Create dictionary of candidate parameters
model_params = {'maxIter': 10
                 ,'rank': 5
                 ,'regParam': 0.1
                 ,'alpha': 100
                 ,'seed': 1234
               }

@app.route('/', methods=['GET'])
def index():
    """Return the main page."""
    colours = ['Red', 'Blue', 'Black', 'Orange']
    rec_data = None
    # Get lists for comic ids and titles
    comics = pd.read_csv('./comrx/webapp/templates/dev_files/top_100_comics.csv')
    ids = comics['comic_id'].tolist()
    titles = comics['comic_title'].tolist()
    comics_dd = comics.loc[:, ['comic_id', 'comic_title', 'img_url']].copy()
    #com_deets = comics_dd.to_dict(orient='dict')
    com_deets = []
    for i, row in comics_dd.iterrows():
        com_deets.append((row['comic_id'],row['comic_title'], row['img_url']))
    
    #dict_urls = dict(com_deets)

    return render_template(
                           'comic_recs.html',
                           colours=colours,
                           titles=titles,
                           rec_data2=rec_data,
                           com_deets=com_deets
                           #,
                           #dict_urls=dict_urls
                           )

# def dropdown():
#     colours = ['Red', 'Blue', 'Black', 'Orange']
#     return render_template('test.html', colours=colours)

#@app.route("/tables")
# @app.route("/")
# def show_tables():
#     data = pd.read_csv('raw_data/fakeqb.csv')
#     # data.set_index(['Name'], inplace=True)
#     # data.index.name=None
#     # females = data.loc[data.Gender=='f']
#     # males = data.loc[data.Gender=='m']
#     # return render_template('view.html',tables=[females.to_html(classes='female'), males.to_html(classes='male')],
#     return render_template('comic_recs.html',tables=[data.to_html(classes='dat')], titles=data.columns.values)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Return a prediction of P(spam)."""
    data = request.json

    # prediction = model.predict_proba([data['user_input']])
    # round_prediction = round(prediction[0][1], 2)
    # return jsonify({'probability': round_prediction})
    # return jsonify({'probability': 'Here are some recommendations.'})
    return jsonify({'probability': data['user_input']})

@app.route('/recommend_old', methods=['GET', 'POST'])
def recommend_old():
    """Return recomendations."""
    data = request.json

    # Interpret string as literal list
    #reading_list = ast.literal_eval(data['comic_input'])
    reading_list = []
    reading_list.append(data['comic_input'])

    # Get Recommendations
    # rec_df = make_comic_recommendations(reading_list=reading_list
    #                                         ,top_n=10
    #                                         ,comics_df=comics_df
    #                                         ,train_data=comics_sold
    #                                         ,model_params=model_params
    #                                         ,spark_instance=spark
    #                                         )

    # top_rec = rec_df.head(1)['comic_title'].values[0]

    # prediction = model.predict_proba([data['user_input']])
    # round_prediction = round(prediction[0][1], 2)
    # return jsonify({'probability': round_prediction})
    # return jsonify({'probability': 'Here are some recommendations.'})
    # return jsonify({'top_rec': top_rec}) 
    #return jsonify({'top_rec': data['comic_input']}) 
    comics = pd.read_csv('./comrx/webapp/templates/dev_files/top_100_comics.csv')
    top_5 = comics.head(5)['comic_title']
    response = make_response(top_5.to_json(orient='records'))
    return response

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    """Return recomendations."""
    data = request.json

    reading_list = []
    reading_list.append(data['comic_input'])

    # Get Recommendations
    rec_df = make_comic_recommendations(reading_list=reading_list
                                            ,top_n=20
                                            ,comics_df=comics_df
                                            ,train_data=comics_sold
                                            ,model_params=model_params
                                            ,spark_instance=spark
                                            )

    comics = pd.read_csv('./comrx/webapp/templates/dev_files/top_100_comics.csv')
    top_5 = comics.tail(5)
    #response = make_response(top_5.to_json(orient='records'))
    response = make_response(rec_df.to_json(orient='records'))

    return response
