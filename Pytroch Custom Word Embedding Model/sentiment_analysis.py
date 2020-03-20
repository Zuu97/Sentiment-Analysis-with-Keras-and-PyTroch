import torch
import numpy as np
import pandas as pd
from variables import*
from torchtext import data
from util import get_data, pad_features
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
from torch import optim as optim

torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

class SentimentAnalysis(object):
    def __init__(self):
        self.train_on_gpu = torch.cuda.is_available()
        if self.train_on_gpu :
            print("running on GPU")
        else:
            print("running on cpu")
        word2index,train_labels,test_labels,train_reviews,test_reviews = get_data()
        Xtrain_pad = pad_features(word2index,train_reviews)
        Xtest_pad  = pad_features(word2index,test_reviews)

        train_labels = torch.LongTensor(train_labels)
        test_labels  = torch.LongTensor(test_labels)
        Xtrain_pad   = torch.LongTensor(Xtrain_pad)
        Xtest_pad    = torch.LongTensor(Xtest_pad)
        print("Train data shape :",Xtrain_pad.shape)
        print("Test data shape  :",Xtest_pad.shape)
        train_data   = TensorDataset(Xtrain_pad, train_labels)
        test_data    = TensorDataset(Xtest_pad, test_labels)

        self.train_loader = DataLoader(
                                      train_data,
                                      shuffle=True,
                                      batch_size=batch_size,
                                      pin_memory=True
                                      )

        self.test_loader  = DataLoader(
                                      test_data,
                                      shuffle=True,
                                      batch_size=batch_size,
                                      pin_memory=True
                                      )

    def sentiment_rnn(self,device):
        self.net = SentimentRNN(device).to(device, non_blocking=True)

    def train_model(self):
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        self.net.train()
        Train_Loss = []
        n_correct = 0
        n_total = 0
        for i in range(num_epochs):
            epoch_loss = 0
            h = self.net.init_hidden(batch_size)
            for inputs, labels in self.train_loader:
                optimizer.zero_grad()
                h = tuple([e.data for e in h])
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                optimizer.zero_grad()
                output, h = self.net(inputs, h)
                loss = loss_function(output.squeeze(), labels.float())
                loss.backward()
                epoch_loss += loss.item()
                n_correct += torch.round(output).eq(labels.float()).sum().item()
                n_total += len(labels)
                nn.utils.clip_grad_norm_(self.net.parameters(), clip)
                optimizer.step()
                train_acc = n_correct/n_total
                torch.cuda.empty_cache()
            Train_Loss.append(epoch_loss)
            print("epoch : {} , train loss : {} train accuracy : {}".format(i+1,round(epoch_loss,3),round(train_acc,3)))
        plt.plot(np.array(Train_Loss))
        plt.show()

    def prediction(self,device):
        with torch.no_grad():
            n_correct = 0
            n_total = 0
            h = self.net.init_hidden(batch_size)
            for batch in self.test_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                h = tuple([e.data for e in h])
                output, h = self.net(inputs, h)
                n_correct += torch.round(output).eq(labels.float()).sum().item()
                n_total += len(labels)
            test_acc = n_correct/n_total
        print("Val_accuracy: {}".format(round(test_acc,3)))

    def save_model(self):
        torch.save(self.net.state_dict(), state_dict)

    def load_model(self,device):
        model = SentimentRNN(device).to(device, non_blocking=True)
        model.load_state_dict(torch.load(state_dict))

class SentimentRNN(nn.Module):
    def __init__(self,device):
        super(SentimentRNN, self).__init__()
        self.output_size=output_size
        self.n_layers=n_layers
        self.hidden_dim=hidden_dim
        self.device = device
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            n_layers,
                            batch_first=True
                            )
        self.dropout = nn.Dropout(keep_prob)
        self.fc = nn.Linear(hidden_dim,hidden_dim)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)
        x = x.long()
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.sig(out)

        out = out.view(batch_size, -1)
        out = out[:,-1]
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device))
        return hidden



if __name__ == "__main__":
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = SentimentAnalysis()
    model.sentiment_rnn(device)
    if not os.path.exists(state_dict):
        model.train_model()
        model.save_model()
    model.load_model(device)
    model.prediction(device)
