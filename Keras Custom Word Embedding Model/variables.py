#Sentiment analysis data
seed = 42
vocab_size = 15000
max_length = 120
embedding_dim = 120
trunc_type = 'post'
oov_tok = "<OOV>"
num_epochs = 10
batch_size = 128
size_lstm1 = 128
size_lstm2 = 64
size_dense = 64
size_output = 1
bias = 0.21600911256083669

#Data paths and weights
train_data_path = 'train.csv'
test_data_path = 'test.csv'
sentiment_path = "model.json"
sentiment_weights = "model.h5"
eclothing_data = 'Womens Clothing E-Commerce Reviews.csv'
preprocessed_eclothing_data = 'Preprocessed Eclothing Data.csv'