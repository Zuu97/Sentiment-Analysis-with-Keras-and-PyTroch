import re
import csv
import random
import os
import numpy as np
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import model_from_json
from sklearn.model_selection import train_test_split
from keras.layers import Input, Embedding, LSTM, Dense, Bidirectional, Dropout
from keras.models import Sequential
from variables import *
from collections import Counter

np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)

class myCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.9975):
            print("\nReached 99.5% train accuracy.So stop training!")
            self.model.stop_training = True

class SentimentAnalyser:
    def __init__(self,Xtrain,Ytrain,Xtest,Ytest):
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.Xtest  = Xtest
        self.Ytest  = Ytest

    def tokenizing_data(self):
        tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
        tokenizer.fit_on_texts(self.Xtrain)

        Xtrain_seq = tokenizer.texts_to_sequences(self.Xtrain)
        self.Xtrain_pad = pad_sequences(Xtrain_seq, maxlen=max_length, truncating=trunc_type)

        Xtest_seq  = tokenizer.texts_to_sequences(self.Xtest)
        self.Xtest_pad = pad_sequences(Xtest_seq, maxlen=max_length)
        self.tokenizer = tokenizer

    def embedding_model(self):

        model = Sequential()
        model.add(Embedding(output_dim=embedding_dim, input_dim=vocab_size, input_length=max_length))
        model.add(Bidirectional(LSTM(size_lstm1,return_sequences=True)))
        model.add(Bidirectional(LSTM(size_lstm2)))
        model.add(Dense(size_dense, activation='relu'))
        model.add(Dense(size_dense, activation='relu'))
        model.add(Dense(size_dense, activation='relu'))
        model.add(Dense(size_output, activation='sigmoid'))

        self.model = model

    def load_model(self):
        json_file = open(sentiment_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(sentiment_weights)
        loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = loaded_model

    def train_model(self,bias):
        callbacks = myCallback()
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()
        class_weights = {1: 1 ,
                        0: 1.6/bias }
        self.model.fit(
            self.Xtrain_pad,
            self.Ytrain,
            batch_size=batch_size,
            epochs=num_epochs,
            validation_data=(self.Xtest_pad,self.Ytest),
            callbacks= [callbacks],
            class_weight=class_weights
            )

    def save_model(self):
        model_json = self.model.to_json()
        with open(sentiment_path, "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(sentiment_weights)

    def predict(self,reviews,labels):
        sequence_data = self.tokenizer.texts_to_sequences(reviews)
        padded_data = pad_sequences(sequence_data, maxlen=max_length)
        if len(padded_data) == 1:
            loss, accuracy = self.model.evaluate(padded_data,np.array([labels]))
        else:
            loss, accuracy = self.model.evaluate(padded_data,labels)
        P = (self.model.predict(padded_data) > 0.5)
        sentiment_score = self.sentiment_score(P)
        print("Val_loss: ",loss)
        print("Val_accuracy: ",accuracy)
        print("sentiment_score: ",sentiment_score)

    def sentiment_score(self,P):
        sentiment_count = Counter([y[0] for y in P])
        positive_count = sentiment_count[1]
        negative_count = sentiment_count[0]
        return positive_count / (positive_count + negative_count)
