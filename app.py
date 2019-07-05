import random
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle
from flask import Flask, request, render_template, jsonify, session, redirect

# Pyspark imports
import pyspark
from pyspark.sql import SparkSession

import ast

import lib.comic_recs as cr

# with open('spam_model.pkl', 'rb') as f:
#     model = pickle.load(f)

app = Flask(__name__, static_url_path="")

# spark config
# spark = SparkSession \
#     .builder \
#     .appName("movie recommendation") \
#     .config("spark.driver.maxResultSize", "1g") \
#     .config("spark.driver.memory", "1g") \
#     .config("spark.executor.memory", "4g") \
#     .config("spark.master", "local[*]") \
#     .getOrCreate()

spark = pyspark.sql.SparkSession.builder.master("local[*]").getOrCreate()

comics_df = spark.read.json('support_data/comics.json')
comics_df.persist()

comics_sold = spark.read.json('raw_data/als_input_filtered.json')
comics_sold.persist()

# Create dictionary of candidate parameters
model_params = {'maxIter': 20
                 ,'rank': 5
                 ,'regParam': 0.1
                 ,'alpha': 100
                 ,'seed': 1234
               }

@app.route('/', methods=['GET'])
def index():
    """Return the main page."""
    colours = ['Red', 'Blue', 'Black', 'Orange']
 
    return render_template(
        'comic_recs.html',
        #'theme.html',
        #'index.html',
        words=['whassup', 'dawg'],
        colours=colours
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

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    """Return recomendations."""
    data = request.json

    # Interpret string as literal list
    reading_list = ast.literal_eval(data['user_input'])

    # Get Recommendations
    rec_df = cr.make_comic_recommendations(reading_list=reading_list
                                            ,top_n=10
                                            ,comics_df=comics_df
                                            ,train_data=comics_sold
                                            ,model_params=model_params
                                            ,spark_instance=spark
                                            )

    top_rec = rec_df.head(1)['comic_title'].values[0]

    # prediction = model.predict_proba([data['user_input']])
    # round_prediction = round(prediction[0][1], 2)
    # return jsonify({'probability': round_prediction})
    # return jsonify({'probability': 'Here are some recommendations.'})
    return jsonify({'top_rec': top_rec}) 
    
