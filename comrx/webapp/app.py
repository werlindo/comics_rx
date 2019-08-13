import pandas as pd
from flask import Flask, request, render_template, make_response

from ..comrx import make_n_comic_recommendations

app = Flask(__name__, static_url_path="")

comic_factors = pd.read_pickle('./comrx/dev/support_data/comics_factors_201908.pkl')


@app.route('/', methods=['GET'])
def index():
    """Return the main page."""
    # Get lists for comic ids and titles
    comics = (pd.read_csv('./comrx/webapp/templates/dev_files/' +
                          'top_100_comics.csv'))
    comics_dd = comics.loc[:, ['comic_id', 'comic_title', 'img_url']].copy()

    com_deets = []
    for i, row in comics_dd.iterrows():
        com_deets.append((row['comic_id'], row['comic_title'], row['img_url']))

    return render_template(
                           'comic_recs.html',
                           com_deets=com_deets
                           )


@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    """Return recomendations."""
    data = request.json

    reading_list = []
    reading_list.append(int(data['comic_input']))
    reading_list.append(int(data['comic_input_2']))
    reading_list.append(int(data['comic_input_3']))

    results = make_n_comic_recommendations(comics=reading_list,
                                           comic_factors=comic_factors,
                                           top_n=int(data['num_recs']))

    response = make_response(results.to_json(orient='records'))
    return response
