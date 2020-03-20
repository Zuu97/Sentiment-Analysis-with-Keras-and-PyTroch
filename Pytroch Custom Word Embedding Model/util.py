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
from variables import  *
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from collections import Counter

vocabulary = []

def get_data():
    global train_data_path, test_data_path
    if not os.path.exists(train_data_path) or not os.path.exists(test_data_path) or not os.path.exists(preprocessed_eclothing_data) :
        df = pd.read_csv(eclothing_data)
        data = df.copy()[['Clothing ID','Review Text','Recommended IND']]
        data = data.dropna(axis = 0, how ='any')
        data['PreProcessed Text'] = data.apply(preprocessed_text_column, axis=1)
        data.to_csv(preprocessed_eclothing_data, encoding='utf-8', index=False)
        print("Upsampling data !!!")
        upsample_data(data)

    if not os.path.exists("word2index"):
        print("Creating word2index")
        df = pd.read_csv(preprocessed_eclothing_data)
        reviews = df['PreProcessed Text']
        word2index = preprocessed_data(reviews)
        preprocessed_data(reviews)

    train_data = pd.read_csv(train_data_path)
    test_data  = pd.read_csv(test_data_path)

    train_idx = np.random.choice(train_data.shape[0], train_size, replace=False)
    test_idx  = np.random.choice(test_data.shape[0], test_size, replace=False)

    train_labels  = np.array(train_data['Recommended IND'],dtype=np.int32)[train_idx]
    test_labels   = np.array(test_data['Recommended IND'],dtype=np.int32)[test_idx]

    train_reviews = np.array(train_data['PreProcessed Text'],dtype='str')[train_idx]
    test_reviews  = np.array(test_data['PreProcessed Text'],dtype='str')[test_idx]

    infile = open("word2index",'rb')
    word2index,_ = pkl.load(infile)
    infile.close()

    print("Data is Ready!!!")
    return word2index,train_labels,test_labels,train_reviews,test_reviews

def lemmatization(lemmatizer,sentence):
    lem = [lemmatizer.lemmatize(k) for k in sentence]
    lem = set(lem)
    return [k for k in lem]

def remove_stop_words(stopwords_list,sentence):
    return [k for k in sentence if k not in stopwords_list]

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

def preprocess_one(review):
    lemmatizer = WordNetLemmatizer()
    tokenizer = RegexpTokenizer(r'\w+')
    stopwords_list = stopwords.words('english')
    review = review.lower()
    remove_punc = tokenizer.tokenize(review) # Remove puntuations
    remove_num = [re.sub('[0-9]', '', i) for i in remove_punc] # Remove Numbers
    remove_num = [i for i in remove_num if len(i)>0] # Remove empty strings
    lemmatized = lemmatization(lemmatizer,remove_num) # Word Lemmatization
    updated_review = remove_stop_words(stopwords_list,lemmatized) # remove stop words
    vocabulary.extend(updated_review)

def preprocessed_data(reviews):
    for review in reviews:
        preprocess_one(review)

    wordcount = Counter(vocabulary)
    sorted_words = wordcount.most_common(len(vocabulary))

    word2index = {w:i+1 for i, (w,c) in enumerate(sorted_words)}

    outfile = open("word2index",'wb')
    pkl.dump((word2index,wordcount), outfile)
    outfile.close()

    return word2index

def tokenize_data(word2index,reviews):
    tokenized_reviews = []
    for review in reviews:
        r = [word2index[w] for w in review.split()]
        tokenized_reviews.append(r)
    return tokenized_reviews

def pad_features(word2index,reviews,max_length=max_length):
    tokenized_reviews = tokenize_data(word2index,reviews)
    padded_data = np.zeros((len(tokenized_reviews), max_length), dtype = int)

    for i, review in enumerate(tokenized_reviews):
        review_len = len(review)

        if review_len <= max_length:
            zeroes = list(np.zeros(max_length-review_len))
            new = zeroes+review
        elif review_len > max_length:
            new = review[0:max_length]

        padded_data[i,:] = np.array(new)
    return padded_data

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
