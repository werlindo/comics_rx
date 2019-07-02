import random
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle
from flask import Flask, request, render_template, jsonify, session, redirect

# with open('spam_model.pkl', 'rb') as f:
#     model = pickle.load(f)

app = Flask(__name__, static_url_path="")

@app.route('/')
def index():
    """Return the main page."""
    return render_template(
        'comic_recs.html',
        #'theme.html',
        #'index.html',
        words=['whassup', 'dawg']
    )

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
    # prediction = model.predict_proba([data['user_input']])
    # round_prediction = round(prediction[0][1], 2)
    # return jsonify({'probability': round_prediction})
    # return jsonify({'probability': 'Here are some recommendations.'})
    return jsonify({'probability': data['user_input']}) 
    
