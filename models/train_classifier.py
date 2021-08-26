import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import pickle

from sklearn.metrics import classification_report

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


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


def tokenize(text):
    
    """
    Tokenizing function, do convert text into tokenzs
    INPUTS:
    text: some text
    
    OUTPUTS:
    text: tokenized text
    """
    
    # Importing modules in the function to avoid problems with parallelization
    import re
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    from nltk.stem.wordnet import WordNetLemmatizer
    
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



def build_model():
    """
    Instantiating a model using sklearn pipelines, then setting a CV grid
    for model selection
    
    INPUT:
    none
    
    OUTPUT:
    pipeline: instantiated pipeline
    """

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier())),
        ])
    
    parameters = [
        {
        'vect__ngram_range': ((1, 1),(1,2)),
        'vect__max_df': (0.5, 1.0),
        'clf__estimator': [RandomForestClassifier()]   
        },
        {'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 1.0),
        'clf__estimator': [KNeighborsClassifier()]
        }
    ]
    

    cv = GridSearchCV(pipeline, param_grid=parameters,n_jobs=-1,scoring='f1_weighted',verbose=1,cv=3)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    

    Parameters
    ----------
    model : the trained model
    X_test : the X_test data from the data split
    Y_test : the Y_test data from the data split
    category_names : the names of the 36 categories

    Returns
    -------
    None.

    """
    # predict on test data
    Y_pred = model.predict(X_test)
    # display results
    for i in range(len(Y_test.columns)):
        #print(classification_report(Y_test.iloc[:,i],Y_pred[:,i]))
        i=i+1
    print(classification_report(Y_test,Y_pred,target_names=category_names))
    pass


def save_model(model, model_filepath):
    """
    
    Parameters
    ----------
    model : the model estimate previously
    model_filepath : the filepath to store the model

    Returns
    -------
    None.

    """
    # save the model to disk
    pickle.dump(model, open(model_filepath, 'wb'))
    pass


def main():
    """
    the main routing to load data, build the model,
    train the model, evaluate and save
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()