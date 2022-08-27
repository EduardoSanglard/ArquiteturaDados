# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 11:28:02 2022

@author: Eduardo Sanglard
"""

from collections import Counter
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from bs4 import BeautifulSoup


subjects = ('Bands','Biomedical','Goats','Sheep')

def tokenize(message: str) -> str:
    
    soup = BeautifulSoup(message, 'html.parser')
    cleaned_text = soup.get_text()
    cleaned_text = cleaned_text.lower()                       # convert to lowercase
    all_words = re.findall("[a-z']+", cleaned_text)   # extract the words
    return set(all_words) 


def msgs_to_data_frame(data: pd.DataFrame) -> pd.DataFrame:
    r = list()
    for idx, row in data.iterrows():
        message = str(row[6])
        rating = row[1]
        words = list(tokenize(message))
        d = {word: 1 for word in words}
        d.update({'Rating': rating, 'Message': message})
        r.append(d)
        
    return(pd.DataFrame(data=r, dtype='float64').fillna(0))

def fix_csv(filePath: str) -> None:
    
    with open(filePath, 'r+') as f:
        
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


def read_data(subject: str) -> pd.DataFrame:
    
    column_names = {0:"FileName",
                    1:"Rating",
                    2:"URL",
                    3:"DateRated",
                    4:"Title"}
    
    # print("\nFiles for "+subject+"\n")
    
    files_path = r".\SW\{0}".format(subject)
    index_path = r".\SW\{0}\index".format(subject)
    
    fix_csv(index_path)
    
    df_index = pd.read_csv(index_path, sep="|", header=None)
    df_index.rename(columns=column_names
                    ,inplace=True)
    
    df_index["Subject"] = subject
    df_index["FileText"] = ''
 
    for index, row in df_index.iterrows():
        
        file_path = files_path + "\\" + str(row[0])
        
        
        with open(file_path, 'r') as f:
            
            full_text = f.read()
            full_text = full_text.strip()
            
            #print("{0} has {1} of size".format(file_path, len(full_text)))
            
            # first_characters = full_text[0: 10]
            # print(first_characters)
            
            row[6] = full_text
    
    return df_index
    

class Classifier:
    
    used_features= []
    gnb = None
    
    def __init__(self):
        self.used_features = []
        self.gnb = GaussianNB()
    
    
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

    
    # Fatorizar Rating
    data['Rating'] = pd.factorize(data['Rating'])[0]

    message_data = msgs_to_data_frame(data)

    X_train, X_test = train_test_split(message_data, test_size=0.3, random_state=1)

    c = Classifier()
    c.train(X_train)
    
    classified = [(msg, rating, c.predict(msg))
                  for msg, rating in X_test[['Message','Rating']].values]

    counts = Counter((rating, predict) for _, rating, predict in classified)

    print(counts.items())

    """

    correctRating = 0
    incorrecRating = 0

    for predicted, target in counts:
        if predicted == target:
            correctRating += counts(predicted, target)
        else:
            incorrecRating += counts(predicted, target)
    
    print(correctRating)
    print(incorrecRating)
    
    """
    
    """
    75 - 25 Split -     GaussianNB()
    
    Total: 82 - 100 % 
    Acertos: 22 - 0.2682926829268293 %
    Erros: 60 - 0.7317073170731707 %
    """
        
if __name__ == "__main__":
    main()