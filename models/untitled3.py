import joblib
import os
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
import pickle
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
print(os.getcwd())
os.chdir('C:/Users/reute/Documents/GitHub/DisasterRepsonseUdacity/models')
print(os.getcwd())
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
model = joblib.load("classifier.pklcls")
print(model.best_params_)

joblib.dump(model.best_index_, "best_estimator.pkl")