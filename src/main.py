import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot
from util import tweet_as_tokens, Embedding, Tweets, load_embedding, load_tweets, split_data


class LogisticRegressionModel(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.num_classes = 2
        self.linear_transform = torch.nn.Linear(self.n_features, self.num_classes)

    def forward(self, x):
        return self.linear_transform(x)
    

def train(features, labels, model, criterion, optimizer, n_epoch):

    for epoch in range(n_epoch):
        #Do a forward pass and calculate the loss
        predictions = model.forward(features)
        loss = criterion(predictions, labels)
        
        #Do a backward pass and a gradient update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print ('Epoch [%d/%d], Loss: %.4f' %(epoch+1, n_epoch, loss.item()))
           
           
def logistic_regression(x_train, y_train, x_test, y_test, n_features, n_epoch, learning_rate):
    
    criterion = torch.nn.CrossEntropyLoss() 
    model_logreg = LogisticRegressionModel(n_features=n_features)
    optimizer = torch.optim.Adam(model_logreg.parameters(), lr=learning_rate)
    
    print("Training...")
    train(x_train, y_train, model_logreg, criterion, optimizer, n_epoch)
    print("Training accuracy:")
    print(accuracy(model_logreg, x_train, y_train))
    print("Testing accuracy:")
    print(accuracy(model_logreg, x_test, y_test))
    

def construct_tensors(emb: Embedding, tweets: Tweets, tweet_count: int):
    assert tweet_count <= len(tweets.pos) and tweet_count <= len(tweets.neg), "Too many tweets"

    x = torch.empty(2 * tweet_count, emb.size)
    y = torch.empty(2 * tweet_count, dtype=torch.long)
    next_i = 0

    for pos, curr_tweets in [(1, tweets.pos), (0, tweets.neg)]:
        for tweet in curr_tweets[:tweet_count]:
            tokens = tweet_as_tokens(tweet, emb.word_dict)
            if len(tokens) == 0:
                continue

            x[next_i, :] = torch.mean(torch.tensor(emb.ws[tokens, :]), dim=0)
            y[next_i] = pos

            next_i = next_i + 1

    # remove excess capacity
    print(f"Dropped {len(x) - next_i} empty tweets")
    x = x[:next_i]
    y = y[:next_i]

    return x, y


def accuracy(model, x, y) -> float:
    model.eval()
    y_pred = model.forward(x)
    y_pred = torch.argmax(y_pred, dim=1)
    return torch.eq(y_pred, y).float().mean()

def main():
    emb = load_embedding("size_200")
    tweets = load_tweets()
    n_epoch = 10000
    learning_rate = 1e-2
    train_ratio = .9
    print("Constructing tensors")
    x, y = construct_tensors(emb=emb, tweets=tweets, tweet_count=100_000)
    n_features = x.shape[1]
    
    # shuffle and split data
    x_train, y_train, x_test, y_test = split_data(x, y, train_ratio)
    #logistic regression
    logistic_regression(x_train, y_train, x_test, y_test, n_features, n_epoch, learning_rate)


if __name__ == '__main__':
    main()
