# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 11:28:02 2022

@author: Eduardo Sanglard
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')

# Declare PorterStemmer and Count Vectorizer to apply on Pipeline
ps = PorterStemmer()
analyzer = CountVectorizer().build_analyzer()

def toStemm(doc):
    return (ps.stem(w) for w in analyzer(doc))


def cleanText(text: str) -> str:
    
    # Remove All HTML Tags with BeautifulSoup
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    text = text.lower().strip()
    
    return text


def read_all_text(file_path: str) -> str:
    with open(file_path, 'r') as f:
        return f.read()


def read_data(subject: str) -> pd.DataFrame:
    
    column_names = {0:"FileName",
                    1:"Rating",
                    2:"URL",
                    3:"DateRated",
                    4:"Title"}
    
    # print(f"Reading files for {subject}")
    
    files_path = r".\SW\{0}".format(subject)
    index_path = r".\SW\{0}\index".format(subject)
    
    #fix_csv(index_path)
    
    df_index = pd.read_csv(index_path, sep="|", header=None)
    df_index.rename(columns=column_names
                    ,inplace=True)
    
    # Generalize Rating (Convert Medium to Cold)
    df_index["Rating"] = df_index["Rating"].apply(lambda r: "cold" if r == "medium" else r)
    
    # Set Subject
    df_index["Subject"] = subject
    
    # Set FilePath    
    df_index["FilePath"] = df_index["FileName"]
    df_index["FilePath"] = df_index["FilePath"].apply(lambda fp: files_path + "\\" + str(fp))
    
    # Read Text Files
    df_index["FileText"] = df_index["FilePath"].apply(lambda fp: read_all_text(fp))
    
    # Remove All HTML Tags from Text
    # df_index["FileText"] = df_index["FileText"].apply(lambda txt: cleanText(txt))
    
    return df_index
    


def main():
    
    subjects = ('Bands', 'BioMedical', 'Goats', 'Sheep')
    #target_names = ['cold', 'medium', 'hot']
    target_names = ['cold', 'hot']
    data = pd.DataFrame()

    for sub in subjects:
        df_index = read_data(sub)
        data = pd.concat([data, df_index])

    # Factorize Rating
    data['Rating'] = pd.factorize(data['Rating'])[0]

    # Show Basic Estatistics and Distribution of Data
    # print(data['Rating'].value_counts())

    pre_process_data_file_path = "data.xlsx"

    # Save Pre process Data
    if os.path.exists(pre_process_data_file_path):
        os.remove(pre_process_data_file_path)
    data.to_excel(pre_process_data_file_path)

    # Split the data for training/test
    X_train, X_test = train_test_split(data, test_size=0.25, random_state=1)

    # Grid Search Pipeline
    
    docs_test = X_test["FileText"]
    
    """
    
    text_clf = Pipeline([
    ('vect', CountVectorizer(analyzer=toStemm)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
    ])
    
    """
    
    text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
    ])
    
    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2)],
        'tfidf__use_idf': (True, False),
        'clf__alpha': (1, 0.1, 1e-2, 1e-3, 1e-4, 1e-5),
        'clf__fit_prior': (True, False)
    }
    
    gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)
    
    gs_clf = gs_clf.fit(X_train["FileText"], X_train["Rating"])

    # Simple test with a string
    target_names[gs_clf.predict(['Rock bands are cirurgical on being cool'])[0]]

    print(f"Best Score: {gs_clf.best_score_}")

    print("Best Parameters:")
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

    predicted = gs_clf.predict(docs_test)
    accuracy = np.mean(predicted == X_test["Rating"])
    
    print(f"\nAccuracy with MultinomialNB Grid Search: {accuracy}")
    
    print(metrics.classification_report(X_test["Rating"], predicted,
    target_names=target_names))


if __name__ == "__main__":
    main()