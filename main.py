# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 11:28:02 2022

@author: Eduardo Sanglard
"""

from collections import Counter
import os, re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from bs4 import BeautifulSoup
# Import of Port Stemmer to stemme word
from nltk.stem import PorterStemmer
import nltk
# from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')

subjects = ('Bands','Biomedical','Goats','Sheep')

def cleanText(text: str) -> str:
    
    # Remove All HTML Tags with BeautifulSoup
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    text = text.lower().strip()
    
    return text
    

def tokenize(message: str) -> str:
    
    bad_words = ("http", "org", "com", "mailto", "href")

    # Use Porter Stemmer
    ps = PorterStemmer() 
    # all_words = re.findall("[a-z'/.&]+", text)
    # all_words = re.findall("[a-z']+", text)

    # Tokenize with NLTK
    tokens = nltk.word_tokenize(message)
    tagged = nltk.pos_tag(tokens)
    
    # Only Nouns
    all_words = [w for w, t in tagged if 'N' in t]
    #all_words = [w for w in tokens]
    
    # Remove words that are numbers or special characters
    all_words = [w for w in all_words if re.match("[a-z']+", w)]

    # Remove bad words
    all_words = [w for w in all_words if not w in bad_words]

    # Steem All Words
    all_words = [ps.stem(w) for w in all_words]
    
    return set(all_words) 


def msgs_to_data_frame(data: pd.DataFrame) -> pd.DataFrame:
    
    r = list()
    
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    
    for idx, row in data.iterrows():
        
        filename = row[0]
        message = row[7]
        rating = row[1]
        subject = row[5]

        # Write Cleaned Text fle
        file_path = r".\Cleaned Text\{0}\{1}.txt".format(subject, filename)
        
        if os.path.exists(file_path):
            os.remove(file_path)
        
        with open(file_path, "w") as txt_file:
            txt_file.write(message)
        
        words = list(tokenize(message))
        d = {word: 1 for word in words}
        d.update({'Rating': rating, 'Message': message})
        r.append(d)
        
    return(pd.DataFrame(data=r).fillna(0))


def fix_csv(filePath: str) -> None:
    """
    
    Parameters
    ----------
    filePath : str
        CSV File Path.

    Returns
    -------
    None
        Fixes the CSV File removing line breaks between rows.

    """
    
    with open(filePath, 'r+', encoding='utf-8') as f:
        
        lines = f.readlines()
        
        lines = [line.strip() for line in lines if line.strip() != ""]
        
        for idx, line in enumerate(lines):
            
            if not '|' in line:   
                if '|' in lines[idx-1]: 
                    lines[idx-1] += " " + line
                else:
                    lines[idx-2] += " " + line
                lines[idx] = ''
        
        lines = [line.strip() for line in lines if line.strip() != ""]
        
        fixedText = "\n".join(lines)
        
        f.seek(0)
        f.write(fixedText)


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
    df_index["FileText"] = df_index["FileText"].apply(lambda txt: cleanText(txt))
    
    # Add the subject int he beggining of the text file
    # df_index["FileText"] = df_index["Subject"] + " " + df_index["FileText"]
    
    return df_index
    

class Classifier:
    
    used_features= []
    gnb = None
    
    def __init__(self):
        self.used_features = []
        #self.gnb = GaussianNB()
        self.gnb = MultinomialNB()
    
    
    def train(self, data):
        self.used_features = data.columns.drop(['Rating', 'Message'])
        self.gnb.fit(data[self.used_features], data['Rating'])
    
    def predict(self, msg):
        words = tokenize(msg)
        words = [w for w in words if w in self.used_features]
        arr = pd.Series(index=self.used_features, dtype='float64').fillna(0)
        arr[words] = 1
        return(self.gnb.predict([arr])[0])


def main():
    
    subjects = ('Bands', 'BioMedical', 'Goats', 'Sheep')
    data = pd.DataFrame()

    for sub in subjects:
        df_index = read_data(sub)
        data = pd.concat([data, df_index])

    # Factorize Rating
    data['Rating'] = pd.factorize(data['Rating'])[0]

    # Show Basic Estatistics and Distribution of Data
    print(data['Rating'].value_counts())

    pre_process_data_file_path = "data.xlsx"
    training_data = "training_data.xlsx"

    # Save Pre process Data
    if os.path.exists(pre_process_data_file_path):
        os.remove(pre_process_data_file_path)
    data.to_excel(pre_process_data_file_path)

    # Perform all data cleaning operations on dataframe
    # message_data = msgs_to_data_frame(data)

    # Split the data for training/test
    X_train, X_test = train_test_split(data, test_size=0.25, random_state=1)

    # Write Training Data in Excel
    if os.path.exists(training_data):
        os.remove(training_data)
    X_train.to_excel(training_data)

    # Generate classifier and train data
    c = Classifier()
    c.train(X_train)
    
    # Gather results (Confusion Matrix)
    classified = [(msg, rating, c.predict(msg))
                  for msg, rating in X_test[['Message','Rating']].values]

    counts = Counter((rating, predict) for _, rating, predict in classified)

    print(counts.items())

if __name__ == "__main__":
    main()