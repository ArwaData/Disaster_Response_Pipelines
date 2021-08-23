import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk import word_tokenize
import gzip
import pickle
import nltk
import re
import os
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
nltk.download(['punkt', 'wordnet'])
from sqlalchemy import *
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.multioutput import MultiOutputClassifier
from sklearn.utils.multiclass import type_of_target
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler, MaxAbsScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV # For optimization
from sklearn.metrics import classification_report,confusion_matrix, precision_score,\
recall_score,accuracy_score,  f1_score,  make_scorer
import sqlite3


def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    connection = engine.connect()
    #con = sqlite3.connect('DisasterResponse.db')
    #cursor = con.cursor()
    #cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    #print("********",cursor.fetchall())
        
    df = pd.read_sql_query('SELECT * FROM InsertTableName',engine)
    X = df.message
    Y = df.loc[:,"related":"direct_report"]
    category_names = Y.columns
    return X,Y,category_names


def tokenize(text):
     # Normalize text
    text = text.lower() 
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) 
    
    
     # Tokenize text
    words = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    
    # Remove stop words
    tokens = [w for w in words if w not in stopwords.words('english')]  

    
    # iterate through each token
    cleanTokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip() # [WordNetLemmatizer().lemmatize(w) for w in tokens]
        cleanTokens.append(clean_tok)
    
    return cleanTokens


def build_model():
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=-1)))
    ])
    
    avg_accuracy_cv = make_scorer(avg_accuracy)
    
    parameters = parameters = {
        
        'clf__estimator__max_depth': [15, 30],  
        'clf__estimator__n_estimators': [100, 250]}
    
    

    cv = GridSearchCV(
        pipeline, 
        param_grid=parameters,
        cv=3,
        scoring=avg_accuracy_cv, 
        verbose=3)
        

    return cv
    
def avg_accuracy(y_test, y_pred):
    
    #This is the score_func used in make_scorer, which would be used in in GridSearchCV 
    
    avg_accuracy=accuracy_score(y_test.values.reshape(-1,1), y_pred.reshape(-1,1))
    
    return avg_accuracy


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    
    for i,col in enumerate(category_names):
        print(col)
        #print(classification_report(np.hstack(Y_test.values.astype(int)), np.hstack(y_pred.values.astype(int))))
        print(classification_report(Y_test, y_pred, target_names=category_names))
        print("***\n")
    pass


def save_model(model, model_filepath):
    
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)
    pass

def main():
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
