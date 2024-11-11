import sys
import pandas as pd
import sqlite3

def load_data(database_filepath):
    # load data from database
    conn = sqlite3.connect('DisasterResponse.db')
    table_name = "disaster_messages"
    df = pd.read_sql(f"SELECT * from {table_name}",conn)
    conn.close()
    return df


def tokenize(text):
    """
    Tokenizes and cleans text by splitting into words, lemmatizing, normalizing case, 
    and removing non-alphanumeric tokens.

    Args:
        text (str): The text string to tokenize and clean.

    Returns:
        list of str: A list of processed, lowercase, lemmatized tokens with punctuation removed.
    """ 
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    # initiate lemmatizer
    clean_tokens = [WordNetLemmatizer().lemmatize(token).lower().strip() for token in tokens if token.isalnum()]
    return clean_tokens






def build_model():
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
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