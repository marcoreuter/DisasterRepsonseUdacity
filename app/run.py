import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine

import os
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
_default_sqlite_db = _project_root / "data/DisasterResponse.db"

SQLALCHEMY_DATABASE_URI = os.environ.get(
    "DATABASE_URL", f"sqlite:///{_default_sqlite_db}"
)


app = Flask(__name__)

def tokenize(text):
    """
    Tokenizing function, do convert text into tokenzs
    INPUTS:
    text: some text
    
    OUTPUTS:
    text: tokenized text
    """
    #Normalizing text
    text=text.lower()
    text=re.sub(r"[^a-zA-z0-9]"," ",text)
    #Tokenize
    text=word_tokenize(text)
    text=[w for w in text if w not in stopwords.words("english")]
    text=[WordNetLemmatizer().lemmatize(w) for w in text]
    text=[WordNetLemmatizer().lemmatize(w,pos='v') for w in text]
    text=[PorterStemmer().stem(w) for w in text]
    return text

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterMessages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    category_names=list(df.iloc[:,4:].columns)
    category_counts = df.iloc[:,4:].sum()
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Disaster Messages',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()