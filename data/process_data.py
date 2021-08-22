# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import fasttext



def load_data(messages_filepath, categories_filepath):
    """
    Load the message and categories data
    
    INPUTS:
    messages_filepath: filepath for the messages data
    categories_filepath: filepath for the categories data
    
    OUTPUTS:
    df: loaded data in a dataframe
    """
    
    # Load messages in csv format
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Drop any duplicates
    messages.drop_duplicates(inplace=True)
    categories.drop_duplicates(inplace=True)
    
    # merge into a single df
    df = messages.merge(categories,how='inner',on='id')

    
    return df


def clean_data(df):
    """
    Clean the loaded data
    
    INPUTS:
    df: any DataFrame
    
    OUTPUTS:
    cleaned_df: cleaned DataFrame
    """
    
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(";",expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames =  [label[:-2] for label in row]
    categories.columns = category_colnames
    
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # see if there are any values that are not 0 or 1 (likely input errors)
    categories=categories[(categories==1) | (categories==0)]
    # drop the original categories column from `df`
    df.drop(['categories'],axis=1,inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    categories['id'] = df.id
    df = df.merge(categories,on='id')
    # drop duplicates
    df.drop_duplicates(inplace=True)
    # remove NaN
    df.dropna(subset=['related'],axis=0,inplace=True)
    # remove any non-english messages
    #pretrained_lang_model = "/home/workspace/data/lid.176.ftz"
    #language_model = fasttext.load_model(pretrained_lang_model)
    #res = df.message.apply(lambda x: language_model.predict(x,k=1)[0][0]=='__label__en')
    #df=df[res==True]
    return df


def save_data(df, database_filename):
    """
    Save data from a DataFrame to a SQL database
    
    INPUTS:
    df: DataFrame to be saved
    database_filename: the database to save the file in
    
    OUTPUTS:
    none
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('DisasterMessages', engine, index=False, if_exists='replace')
    pass  


def main():
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