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
import matplotlib.pyplot as plt

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

    if not os.path.exists(word2vec_path):
        print("Creating word2vector mapping")
        df = pd.read_csv(preprocessed_eclothing_data)
        reviews = df['PreProcessed Text']
        word2vector()

    train_data = pd.read_csv(train_data_path)
    test_data  = pd.read_csv(test_data_path)

    train_idx = np.random.choice(train_data.shape[0], train_size, replace=False)
    test_idx  = np.random.choice(test_data.shape[0], test_size, replace=False)

    train_labels  = np.array(train_data['Recommended IND'],dtype=np.int32)[train_idx]
    test_labels   = np.array(test_data['Recommended IND'],dtype=np.int32)[test_idx]

    train_embeddings = np.load(train_embs) if os.path.exists(train_embs) else word_embedding(train_data)
    test_embeddings  = np.load(train_embs) if os.path.exists(test_embs) else word_embedding(test_data,False)

    train_embeddings = train_embeddings[train_idx]
    test_embeddings = train_embeddings[test_idx]
    print("Embeddings are Ready!!!")
    return train_labels,test_labels,train_embeddings,test_embeddings

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

def word2vector():
    word2vec = {}
    with open(glove_path, "r" , encoding="utf8") as lines:
        for line in lines:
            word = line.split()[0]
            vector = line.split()[1:]
            vector = list(map(float, vector))
            word2vec[word] = vector
    outfile = open(word2vec_path,'wb')
    pkl.dump(word2vec, outfile)
    outfile.close()

def pad_review(tokens):
    review_len = len(tokens)
    if review_len <= max_length:
        padded = [pad_token]*(max_length-review_len)
        new_tokens = tokens+padded
    elif review_len > max_length:
        new_tokens = tokens[0:max_length]
    return new_tokens

def word_embedding(df, Train=True):
    infile = open(word2vec_path,'rb')
    word2vec = pkl.load(infile)
    infile.close()

    reviews = df['PreProcessed Text']
    N = len(reviews)
    all_embeddings = np.zeros((N,max_length,embedding_dim), dtype=float)
    for i,review in enumerate(reviews):
        tokens = review.split(' ')
        tokens = pad_review(tokens)
        for j,token in enumerate(tokens):
            if token in word2vec:
                vec = word2vec[token]
                all_embeddings[i,j,:] = vec
            elif token == pad_token:
                pass
            else:
                vec = word2vec[unknown_token]
                all_embeddings[i,j,:] = vec
    if Train:
        print("Save training embeddings")
        np.save(train_embs,all_embeddings)
    if not Train:
        print("Save test embeddings")
        np.save(test_embs,all_embeddings)

    return all_embeddings
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