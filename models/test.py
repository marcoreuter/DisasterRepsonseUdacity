# -*- coding: utf-8 -*-
import joblib
"""
Created on Sun Aug 22 14:06:57 2021

@author: reute
"""

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


model = joblib.load("classifier.pkl")

print(model.refit_time_)
