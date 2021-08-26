# Load Packages
import pandas as pd
from sqlalchemy import create_engine
from wordcloud import WordCloud


def load_data(database_filepath):
    """
    Load data from SQL database to use for modelling
    
    INPUTS:
    database_filepath: filepath of the database
    
    OUTPUTS:
    X: dataframe of explanatory variables
    y: dataframe of dependant variables
    category_names: names of the categories of the dependant variables 
    """
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterMessages',engine)
    X = df.message
    y = df.iloc[:,4:]
    category_names=y.columns
    #X = X.head(1000)
    #y = y.head(1000)
     
    return X, y, category_names


# Load data
X,y,category_names = load_data("data/DisasterResponse.db")
# Generate Wordcloud
wordcloud = WordCloud(width=1000,height=400).generate(' '.join(X))
# Generate plot
wordcloud.to_file("/app/static/wordcloud.png")
