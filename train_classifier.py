# import libraries
import numpy as np
from tqdm import tqdm
import sqlite3
import pandas as pd
import seaborn as sns
import sys
import re
import pickle
import warnings
from nltk.tokenize import word_tokenize
from sklearn.model_selection import cross_val_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings('ignore')

def load_data(database_filepath):
    """
    Loads data from the specified SQLite database filepath and returns the data as a DataFrame.

    Args:
        database_filepath (str): The file path to the SQLite database.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded data.
    """
    conn = sqlite3.connect(database_filepath)
    table_name = "disaster_messages"
    df = pd.read_sql(f"SELECT * from {table_name}", conn)
    conn.close()
    X = df['message']  # Features
    y = df.drop(columns=['id', 'message', 'genre'])  # Targets
    return X,y



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






# Define build_model function
def build_model(database_filepath, module=1):
    """
    Builds and returns one of three machine learning pipelines based on the specified module:
    a standard RandomForest pipeline, a GridSearchCV with RandomForest pipeline, or a Logistic Regression pipeline for multi-output classification.

    Args:
        database_filepath (str): The file path to the SQLite database.
        module (int, optional): Specifies which model pipeline to return.
                                - 1: Standard RandomForest pipeline
                                - 2: GridSearchCV with RandomForest pipeline
                                - 3: Logistic Regression pipeline
                                Default is 1.

    Returns:
        Pipeline or GridSearchCV: The selected machine learning pipeline.
                                  - Standard RandomForest pipeline (module=1)
                                  - GridSearchCV with RandomForest (module=2)
                                  - Logistic Regression pipeline (module=3)
    """
    #df = load_data(database_filepath)
    #X = df['message']  
    #y = df.drop(columns=['id', 'message', 'genre'])  
    X,y = load_data(database_filepath)
    # Standard RandomForest Pipeline
    standard_pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),  
        ('tfidf', TfidfTransformer()),                  
        ('clf', MultiOutputClassifier(RandomForestClassifier()))   ])
    
    # GridSearchCV with RandomForest Pipeline
    grid_search_pipeline = GridSearchCV(
        standard_pipeline,
        param_grid={
            'vect__max_df': [0.75, 1.0],  
            'tfidf__use_idf': [True, False],  
            'clf__estimator__n_estimators': [50, 100],  
            'clf__estimator__max_depth': [None, 10, 20]  
        },
        cv=3,
        verbose=2,
        n_jobs=-1
    )
    
    # Logistic Regression Pipeline
    logistic_pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),  
        ('tfidf', TfidfTransformer()),                 
        ('clf', MultiOutputClassifier(LogisticRegression(solver='lbfgs',\
                                                          max_iter=1000)))  
    ])
    if module ==1:
        return standard_pipeline
    elif module ==2:
         return grid_search_pipeline 
    else: 
        return logistic_pipeline



def evaluate_model(model, X_test, y_test):
    """
    Evaluates the provided model on test data and prints classification reports
    for each target category in y_test.

    Args:
        model: The trained model to evaluate.
        X_test (DataFrame): Test features.
        y_test (DataFrame): True labels for each target category.

    Returns:
        None
    """
    y_pred = model.predict(X_test)

    # Loop through each column in y_test and print classification report
    for i, column in enumerate(y_test.columns):
        print(f"Category: {column}")
        print(classification_report(y_test.iloc[:, i], y_pred[:, i], zero_division=0))


def save_model(model, model_filepath):
    """
    Saves the provided model to a specified file path using pickle.

    Args:
        model: The trained model to save.
        model_filepath (str): The file path where the model should be saved.

    Returns:
        None
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)





def main():
    if len(sys.argv) == 4:  # Expecting 4 arguments (including module)
        database_filepath, model_filepath, module = sys.argv[1:]

        try:
            module = int(module)
            if module not in [1, 2, 3]:
                raise ValueError("Module must be 1, 2, or 3.")
        except ValueError as e:
            print(f"Invalid module argument: {e}")
            sys.exit(1)

    elif len(sys.argv) == 3:  # Expecting 3 arguments (without module)
        database_filepath, model_filepath = sys.argv[1:]
        module = None  # No module argument
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')
        sys.exit(1)

    print(f'Loading data...\n    DATABASE: {database_filepath}')
    X, Y = load_data(database_filepath)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    print('Building model...')
    model = build_model(database_filepath)

    print('Training model...')
    for _ in tqdm(range(1), desc="Training model"):
        model.fit(X_train, Y_train)

    print('Model training complete!')

    print('------------------------')
    print('Evaluating model...')
    evaluate_model(model, X_test, Y_test)

    print(f'Saving model...\n    MODEL: {model_filepath}')
    save_model(model, model_filepath)

    print('Trained model saved!')

if __name__ == '__main__':
    main()
