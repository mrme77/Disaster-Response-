import json
import plotly
import pandas as pd
import joblib
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
#from sklearn.externals import joblib
import sqlite3


app = Flask(__name__)


def tokenize(text):
    """
    Tokenizes and cleans text by splitting into words, lemmatizing, normalizing case, 
    and removing non-alphanumeric tokens.

    Args:
        text (str): The text string to tokenize and clean.

    Returns:
        list of str: A list of processed, lowercase, lemmatized tokens with punctuation removed.
    """ 
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    clean_tokens = [WordNetLemmatizer().lemmatize(token).lower().strip() for token in tokens if token.isalnum()]
    return clean_tokens


# def tokenize(text):
#     tokens = word_tokenize(text)
#     lemmatizer = WordNetLemmatizer()

#     clean_tokens = []
#     for tok in tokens:
#         clean_tok = lemmatizer.lemmatize(tok).lower().strip()
#         clean_tokens.append(clean_tok)

#     return clean_tokens

# load data from db
conn = sqlite3.connect('DisasterResponse.db')
table_name = "disaster_messages"
df = pd.read_sql(f"SELECT * from {table_name}",conn)


model = joblib.load("/Users/pasqualesalomone/Desktop/UdacityNotes/DataEngineering/de_project/Disaster-Response-/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # data preview and visualizations
    table_html = df.head(3).to_html(classes='table table-striped')
    
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    disaster_columns = ['electricity', 'earthquake', 'storm','floods','fire','hospitals']  
    disaster_counts = df[disaster_columns].sum()  
    disaster_names = disaster_counts.index.tolist()  

    graphs = [
        # Bar chart for genre distribution
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Genre"}
            }
        },

        # Bar chart for some of disaster messages types (electricity, earthquake, storm)
        {
            'data': [
                Bar(
                    x=disaster_names,
                    y=disaster_counts                )
            ],
            'layout': {
                'title': 'Top 10 Disaster Message Types',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Disaster Type"}
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html',table_html=table_html,ids=ids, graphJSON=graphJSON)


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