import sys
import nltk
nltk.download(['punkt', 'wordnet'])

import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from string import punctuation
import pickle
import sqlite3

def load_data(database_filepath):
     """
    Load data from a SQLite database.

    Arguments:
        database_filepath (str): Filepath to the SQLite database.

    Output:
        X= A Series containing the messages.
        Y= A DataFrame containing the target categories.
        category_names= List of category names.
    """
    # create engine and bring in database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterResponse', engine)
    # Define X and Y variables
    X = df['message']
    Y = df.drop(['original','genre','message','id'], axis=1)
    Y= Y.astype(int)
    category_names = list(df.columns[4:])
    return X, Y, category_names 
    

def tokenize(text):
    """
    Tokenize and lemmatize a text.

    Arguments:
        text= Input text.

    Output:
        clean_tokens= List of tokenized and lemmatized words.
    """
    # remove punctations
    text =  ''.join([c for c in text if c not in punctuation])
    
    #tokenize text
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    # clean tokens 
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Builds a machine learning pipeline.

    Output:
        Model= Machine learning model.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters= {
        'clf__estimator__n_estimators': [8, 15],
        'clf__estimator__min_samples_split': [2],
    }
    # new model with improved parameters
    model = GridSearchCV(estimator=pipeline, param_grid=parameters, cv=3)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
     """
    Evaluate the model and print classification reports.

    Arguments:
        model= Trained machine learning model.
        X_test= Test set of data.
        Y_test= Test set of data.
        category_names= List of category names.
        
        Output:
        none
    """
    y_pred = model.predict(X_test)
    for i, col in enumerate(Y_test):
        print(col)
        print(classification_report(Y_test[col], y_pred[:, i],                                                             target_names=category_names))

def save_model(model, model_filepath):
     """
    Save the trained model to a pickle file.

    Arguments:
        model= Trained model.
        model_filepath= Filepath to save the model.
    """
    with open('classifier.pkl', 'wb') as f:
        pickle.dump(model, f)

    


def main():
    '''
    Runs all the steps to classify data
    '''
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