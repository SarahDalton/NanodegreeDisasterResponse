import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Loads datasets and merges them both
    
    Arguments:
        messages_filepath = Path to the CSV file containing messages
        categories_filepath = Path to the CSV file containing categories
    Output:
        df = Merged dataframe of messages and categories
    
    """
    # read in the datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge the datasets
    df = categories.merge(messages, how='left', on=['id'])    
    return df


def clean_data(df):
    '''
    Cleans the input dataframe by splitting the text data
    in the "categories" column into separate columns, 
    then converts category values to binary values
    and removes duplicates.    
    Arguments: 
        df= input dataframe from load function
        
    Output: 
        df= cleaned dataframe    
    '''
    # split text data from 'categories' column into seperate columns
    categories =  df['categories'].str.split(';', expand=True)
    row = categories[0:1]
    category_colnames = row.apply(lambda x: x.str[:-2]).values.tolist()
    
    categories.columns = category_colnames
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
    
    # convert column from string to numeric
    categories[column] = pd.to_numeric(categories[column])
    df= df.drop('categories', axis=1)
    # merge the new columns with the original df
    df = pd.concat([df, categories], axis=1)
    # drop duplicates
    df=df.drop_duplicates()
    return df


def save_data(df, database_filename):
    '''
    Saves and exports the input dataframe into a sqlite database.
    
    Arguments:
        df= input dataframe
        database_filename= file path and name for the database
    Output: 
        None
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('DisasterResponse', engine, if_exists = 'replace', index=False)
      


def main():
    '''
    Runs all the steps to process data
    '''
    
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
