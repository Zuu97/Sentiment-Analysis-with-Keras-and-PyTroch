import re
import os
import pandas as pd
import numpy as np
import pickle as pkl
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.utils import resample
from sklearn.utils import shuffle
from variables import train_data_path, test_data_path, preprocessed_eclothing_data, eclothing_data
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from collections import Counter

def get_sentiment_data():
    global train_data_path, test_data_path
    if not os.path.exists(train_data_path) or not os.path.exists(test_data_path) or not os.path.exists(preprocessed_eclothing_data):
        print("Upsampling data !!!")
        df = pd.read_csv(eclothing_data)
        data = df.copy()[['ID','Clothing ID','Review Text','Recommended IND']]
        data['PreProcessed Text'] = data.apply(preprocessed_text_column, axis=1)
        data.to_csv(preprocessed_eclothing_data, encoding='utf-8', index=False)
        upsample_data(data)
    train_data = pd.read_csv(train_data_path)
    test_data  = pd.read_csv(test_data_path)

    train_labels  = np.array(train_data['Recommended IND'],dtype=np.int32)
    test_labels   = np.array(test_data['Recommended IND'],dtype=np.int32)

    train_reviews = np.array(train_data['PreProcessed Text'],dtype='str')
    test_reviews  = np.array(test_data['PreProcessed Text'],dtype='str')
    print("Data is Ready!!!")
    return train_labels,test_labels,train_reviews,test_reviews

def lemmatization(lemmatizer,sentence):
    lem = [lemmatizer.lemmatize(k) for k in sentence]
    lem = set(lem)
    return [k for k in lem]

def remove_stop_words(stopwords_list,sentence):
    return [k for k in sentence if k not in stopwords_list]

def preprocess_one(review):
    lemmatizer = WordNetLemmatizer()
    tokenizer = RegexpTokenizer(r'\w+')
    stopwords_list = stopwords.words('english')
    review = review.lower()
    remove_punc = tokenizer.tokenize(review) # Remove puntuations
    remove_num = [re.sub('[0-9]', '', i) for i in remove_punc] # Remove Numbers
    remove_num = [i for i in remove_num if len(i)>0] # Remove empty strings
    lemmatized = lemmatization(lemmatizer,remove_num) # Word Lemmatization
    remove_stop = remove_stop_words(stopwords_list,lemmatized) # remove stop words
    updated_review = ' '.join(remove_stop)
    return updated_review

def preprocessed_text_column(row):
    review = str(row['Review Text'])
    lemmatizer = WordNetLemmatizer()
    tokenizer = RegexpTokenizer(r'\w+')
    stopwords_list = stopwords.words('english')
    review = review.lower()
    remove_punc = tokenizer.tokenize(review) # Remove puntuations
    remove_num = [re.sub('[0-9]', '', i) for i in remove_punc] # Remove Numbers
    remove_num = [i for i in remove_num if len(i)>0] # Remove empty strings
    lemmatized = lemmatization(lemmatizer,remove_num) # Word Lemmatization
    remove_stop = remove_stop_words(stopwords_list,lemmatized) # remove stop words
    updated_review = ' '.join(remove_stop)
    return updated_review

def upsample_data(data):
    data_majority = data[data['Recommended IND'] == 1]
    data_minority = data[data['Recommended IND'] == 0]

    # bias = data_minority.shape[0]/data_majority.shape[0]

    # lets split train/test data first then
    train = pd.concat([data_majority.sample(frac=0.8,random_state=200),
            data_minority.sample(frac=0.8,random_state=200)])
    test = pd.concat([data_majority.drop(data_majority.sample(frac=0.8,random_state=200).index),
            data_minority.drop(data_minority.sample(frac=0.8,random_state=200).index)])

    train = shuffle(train)
    test = shuffle(test)

    print('positive data in training:',(train['Recommended IND'] == 1).sum())
    print('negative data in training:',(train['Recommended IND'] == 0).sum())
    print('positive data in test:',(test['Recommended IND'] == 1).sum())
    print('negative data in test:',(test['Recommended IND'] == 0).sum())

    # Separate majority and minority classes in training data for up sampling
    data_majority = train[train['Recommended IND'] == 1]
    data_minority = train[train['Recommended IND'] == 0]

    print("majority class before upsample:",data_majority.shape)
    print("minority class before upsample:",data_minority.shape)

    # Upsample minority class
    data_minority_upsampled = resample(data_minority,
                                      replace=True,     # sample with replacement
                                      n_samples= data_majority.shape[0],    # to match majority class
                                      random_state=42) # reproducible results

    # Combine majority class with upsampled minority class
    train_data_upsampled = pd.concat([data_majority, data_minority_upsampled])

    # Display new class counts
    print("After upsampling\n",train_data_upsampled['Recommended IND'].value_counts(),sep = "")
    train_data_upsampled = shuffle(train_data_upsampled)

    train_data_upsampled = train_data_upsampled.dropna(axis = 0, how ='any')
    test = test.dropna(axis = 0, how ='any')
    train_data_upsampled.to_csv(train_data_path, encoding='utf-8', index=False)
    test.to_csv(test_data_path, encoding='utf-8', index=False)

def preprocessed_data(reviews):
    updated_reviews = []
    if isinstance(reviews, np.ndarray) or isinstance(reviews, list):
        for review in reviews:
            updated_review = preprocess_one(review)
            updated_reviews.append(updated_review)
    elif isinstance(reviews, np.str_)  or isinstance(reviews, str):
        updated_reviews = [preprocess_one(reviews)]

    return np.array(updated_reviews)

def get_reviews_for_id():
    data = pd.read_csv(preprocessed_eclothing_data)
    cloth_ids = data['Clothing ID']
    while True:
        cloth_id = int(input("Enter cloth Id :"))
        if cloth_id < max(cloth_ids) + 1:
            cloth_id_data = data[data['Clothing ID'] == cloth_id]
            reviews = cloth_id_data['PreProcessed Text']
            labels = cloth_id_data['Recommended IND']
            break
        print("Please enter cloth ID below 1206 !")

    return reviews.to_numpy(), labels.to_numpy()
