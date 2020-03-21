#Sentiment analysis data
vocab_size = 13000
max_length = 56
embedding_dim = 100
num_epochs = 50
clip=5
batch_size = 100
n_layers = 2
hidden_dim = 64
learning_rate = 0.001
keep_prob = 0.3
output_size = 1
train_size = 28000
test_size = 3000
unknown_token = 'unk' # OOV token replace by unk word vector
pad_token = '_pad_' # while padding add _pad_ word to make the sequence to get maximum length

#Data paths and weights
train_embs = "train_embeddings.npy"
test_embs = "test_embeddings.npy"
glove_path = "glove.6B.100d.txt"
word2vec_path = "word2vec"
train_data_path = 'train.csv'
test_data_path = 'test.csv'
sentiment_path = "model.json"
sentiment_weights = "model.h5"
eclothing_data = 'Womens Clothing E-Commerce Reviews.csv'
preprocessed_eclothing_data = 'Preprocessed Eclothing Data.csv'
state_dict = "model_dict.pt"