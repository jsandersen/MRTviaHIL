from enum import Enum
import os
import random
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import reuters 

def _load_imdb(data_dir = './aclImdb/'):
    X = []
    y = []
    for partition in ["train", "test"]:
        for category  in ["pos", "neg"]:
            lable = 0 if category  == "neg" else 1

            path = os.path.join(data_dir, partition, category )
            files = os.listdir(path)
            for f_name in files:
                with open(os.path.join(path, f_name), "r") as f:
                    review = f.read()
                    X.append(review)
                    y.append(lable)

    return X, y

def _load_app_store(data_dir = './dataset.csv'):
    df = pd.read_csv(data_dir)
    
    df_en = df[(df.lang == 'en') & (df.source == 'app_review') & ~df['text'].isna()]
    
    X = df_en['text'].to_list()
    y = df_en['category'].map({'inq': 0, 'pbr': 1, 'irr': 2}).to_list()
    
    return X, y
    

def _load_hate_speach(data_dir = './labeled_data.csv'):
    nRowsRead = None
    df0 = pd.read_csv(data_dir, delimiter=',', nrows = nRowsRead)
    df0.dataframeName = data_dir
    nRow, nCol = df0.shape
    c=df0['class']
    df0.rename(columns={'tweet' : 'text', 'class' : 'category'}, inplace=True)
    a=df0['text']
    b=df0['category'].map({0: 'hate_speech', 1: 'offensive_language', 2: 'neither'})
    df= pd.concat([a,b,c], axis=1)
    
    X = df['text'].values
    y = df['class'].values
    return X, y

cat_8 = ['acq', 'crude', 'earn', 'grain', 'interest', 'money-fx', 'ship', 'trade']

def _load_reuters(data_dir = None):
    nltk.download('reuters')
    
    dic_map = {}
    for i in range(len(cat_8)):
        dic_map[cat_8[i]] = i

    documents = reuters.fileids()
    train_docs = list(filter(lambda doc: doc.startswith("train"), documents));
    test_docs = list(filter(lambda doc: doc.startswith("test"), documents));
    
    def _get_data(docs):
        docs_text = []
        docs_label = []
        for index, i in  enumerate(docs):
            docs_text.append(reuters.raw(fileids=[i]))
            docs_label.append(reuters.categories(i))

        X = []
        y = []
        for cat in cat_8:
            count = 0
            for i in range(len(docs_label)): 
                if cat in docs_label[i]:
                    if len(docs_label[i]) > 1:
                        double = 0
                        for label in docs_label[i]:
                            if label in cat_8:
                                double += 1
                        if double > 1:
                            continue;
                    count += 1
                    X.append(docs_text[i])
                    y.append(cat)
        return X, y
    
    X_train, y_train = _get_data(train_docs)
    X_test, y_test = _get_data(test_docs)
    
    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])
    y = [dic_map[i] for i in y]
    
    return X, y
   
                
def _load_not_yet_implementet():
    print('not yet implementet')
    
    
class Dataset(Enum):
    IMDB = 0
    APP_STORE = 1
    REUTERS = 2
    HATE_SPEECH = 3

def load_data(dataset, data_dir=None):
    assert type(dataset) == Dataset, "Invalid Argument"
    
    if dataset is Dataset.IMDB:
        return _load_imdb(data_dir)
    elif dataset is Dataset.APP_STORE:
        return _load_app_store(data_dir)
    elif dataset is Dataset.HATE_SPEECH:
        return _load_hate_speach(data_dir)
    elif dataset is Dataset.REUTERS:
        return _load_reuters()
     

    
    