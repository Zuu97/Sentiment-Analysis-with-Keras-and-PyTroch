#Sentiment analysis data
vocab_size = 13000
max_length = 56
embedding_dim = 100
num_epochs = 30
clip=5
batch_size = 100
n_layers = 2
hidden_dim = 64
learning_rate = 0.001
keep_prob = 0.3
output_size = 1
train_size = 28000
test_size = 3000

#Data paths and weights
train_data_path = 'train.csv'
test_data_path = 'test.csv'
sentiment_path = "model.json"
sentiment_weights = "model.h5"
eclothing_data = 'Womens Clothing E-Commerce Reviews.csv'
preprocessed_eclothing_data = 'Preprocessed Eclothing Data.csv'
state_dict = "model_dict.pt"