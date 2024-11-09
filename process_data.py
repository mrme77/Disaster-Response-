import sys
import pandas as pd
import sqlite3


def load_data(messages_filepath, categories_filepath):
    """
    Load and merge messages and categories datasets.

    Args:
        messages_filepath (str): Filepath for the CSV file containing messages 
        data.
        categories_filepath (str): Filepath for the CSV file containing 
        categories data.

    Returns:
        pd.DataFrame: A DataFrame containing the merged content of messages 
        and categories,joined on the 'id' column.
                      
    The resulting DataFrame will contain all columns from both datasets, with 
    any overlapping entries in the 'id' column serving as the basis for the
    merge.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on ='id')
    return df

def clean_data(df):
    """
    Cleans and preprocesses a DataFrame containing disaster message data by 
    expanding the 'categories' column into individual binary category columns 
    and removing duplicates.

    This function performs the following steps:
        - Splits the 'categories' column into separate columns, one for each 
          category.
        - Sets the column names for each category using the first row of data.
        - Converts category values from strings to integers.
        - Removes the original 'categories' and 'original' columns from the 
          DataFrame.
        - Removes duplicate entries based on the 'id' column.
        - Resets the DataFrame index.

    Args:
        df (pd.DataFrame): The DataFrame containing disaster message data, 
                           including a 'categories' column with multi-label 
                           categorical data.

    Returns:
        pd.DataFrame: A cleaned DataFrame with expanded category columns, 
        deduplicated entries,and updated index.
    """
    categories_split = df['categories'].str.split(';', expand=True)
    category_colnames = categories_split.iloc[0].apply(
        lambda x: x.split('-')[0])
    categories_split.columns = category_colnames
    categories_split = categories_split.apply(
        lambda col: col.str.split('-').str[1].astype(int)
    )
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories_split], axis=1)
    df.drop('original', axis=1, inplace=True)    
    df.drop_duplicates(subset='id',inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df




def save_data(df, database_filename):
    """
    Saves a DataFrame to a SQLite database file with a specified table name.

    This function connects to a SQLite database, saves the DataFrame as a 
    table named 'disaster_messages', and closes the connection.

    Args:
        df (pd.DataFrame): The DataFrame to save to the SQLite database.
        database_filename (str): The path to the SQLite database file where 
        the DataFrame will be stored.

    Returns:
        None
    """
    conn = sqlite3.connect(database_filename)
    df.to_sql('disaster_messages', conn, if_exists='replace', index=False)
    # conn.commit() is unnecessary because to_sql commits changes automatically 
    #  when saving to SQLite.
    conn.close()
    return None  


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