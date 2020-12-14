import numpy as np
import torch
from matplotlib import pyplot

from util import tweet_as_tokens, Embedding, Tweets, load_embedding, load_tweets, split_data


def train(model, x_train, y_train, x_test, y_test, loss_func, optimizer, epochs: int, batch_size: int, device: str):
    losses = np.zeros(epochs // 10)
    train_accs = np.zeros(epochs // 10)
    test_accs = np.zeros(epochs // 10)

    for epoch in range(epochs):
        model.train()

        shuffle = torch.randperm(len(x_train), device=device)
        batch_count = len(x_train) // batch_size

        epoch_loss = 0
        epoch_acc = 0

        for b in range(batch_count):
            batch_i = shuffle[b * batch_size: (b + 1) * batch_size]
            x_train_batch = x_train[batch_i]
            y_train_batch = y_train[batch_i]

            predictions = model.forward(x_train_batch)
            loss = loss_func(predictions, y_train_batch)

            epoch_loss += loss.item() / batch_count
            epoch_acc += accuracy(predictions, y_train_batch) / batch_count

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        y_test_pred = model.forward(x_test)
        test_acc = accuracy(y_test_pred, y_test)
        print(f'Epoch {epoch}/{epochs}, loss {epoch_loss:.4f} acc {epoch_acc:.4f} test_acc {test_acc:.4f}')

        losses[epoch // 10] = epoch_loss
        train_accs[epoch // 10] = epoch_acc
        test_accs[epoch // 10] = test_acc

    return losses, train_accs, test_accs

def train1(model, x_train, y_train, z_train, loss_func, optimizer, epochs: int, batch_size: int, device: str):
    losses = np.zeros(epochs // 10)
    train_accs = np.zeros(epochs // 10)
    test_accs = np.zeros(epochs // 10)

    for epoch in range(epochs):
        model.train()

        shuffle = torch.randperm(len(x_train), device=device)
        batch_count = len(x_train) // batch_size

        epoch_loss = 0
        epoch_acc = 0

        for b in range(batch_count):
            batch_i = shuffle[b * batch_size: (b + 1) * batch_size]
            x_train_batch = x_train[batch_i]
            y_train_batch = y_train[batch_i][:,0]
            z_train_batch = z_train[batch_i]
            #print(x_train_batch.size())
            #print(y_train_batch)
            predictions = model.forward(x_train_batch)
            
            #print(predictions)
            #print(predictions)
            #print(predictions.size())
            
            loss = loss_func(predictions, y_train_batch)
            #loss = torch.sum(loss)
            loss = torch.sum(torch.mul(loss,z_train_batch))
           # print(loss)
            
            
            epoch_loss += loss.item() / batch_count
            epoch_acc += accuracy(predictions, y_train_batch) / batch_count

            optimizer.zero_grad()
            loss.backward( )
            optimizer.step()

        #y_test_pred = model.forward(x_test)
        #test_acc = accuracy(y_test_pred, y_test)
        print(f'Epoch {epoch}/{epochs}, loss {epoch_loss:.4f} acc {epoch_acc:.4f}')

        losses[epoch // 10] = epoch_loss
        train_accs[epoch // 10] = epoch_acc
        

    return losses, train_accs


def train2(model, x_train, y_train, z_train, loss_func, optimizer, epochs: int, batch_size: int, device: str):
    losses = np.zeros(epochs // 10)
    train_accs = np.zeros(epochs // 10)
    test_accs = np.zeros(epochs // 10)
    
    totrain = torch.randn(len(x_train),2)
    print(totrain.size())
    
    for epoch in range(epochs):
        model.train()

        shuffle = torch.randperm(len(totrain), device=device)
        batch_count = len(totrain) // batch_size

        epoch_loss = 0
        epoch_acc = 0

        for b in range(batch_count):
            batch_i = shuffle[b * batch_size: (b + 1) * batch_size]
            #x_train_batch = x_train[batch_i]
            y_train_batch = y_train[batch_i][:,0]
            z_train_batch = z_train[batch_i]
            totrain_batch = totrain[batch_i]
            #print(x_train_batch.size())
            #print(y_train_batch)
            predictions = model.forward(totrain_batch)
            
            #print(predictions)
            #print(predictions)
            #print(predictions.size())
            
            loss = loss_func(predictions, y_train_batch)
            loss = torch.sum(torch.mul(loss,z_train_batch))
           # print(loss)
            
            
            epoch_loss += loss.item() / batch_count
            epoch_acc += accuracy(predictions, y_train_batch) / batch_count

            optimizer.zero_grad()
            loss.backward( )
            optimizer.step()

        #y_test_pred = model.forward(x_test)
        #test_acc = accuracy(y_test_pred, y_test)
        print(f'Epoch {epoch}/{epochs}, loss {epoch_loss:.4f} acc {epoch_acc:.4f}')

        losses[epoch // 10] = epoch_loss
        train_accs[epoch // 10] = epoch_acc
        

    return losses, train_accs
    
def test(emb:Embedding, tweets : Tweets, tweet_count :int, epochs: int, batch_size: int, learning_rate: int, device: str, model, loss_func, optimizer):
    
    losses = np.zeros(epochs // 10)
    test_accs = 0
    epoch_acc = 0
    
    L = 2*tweet_count
    
    print('L')
    print(L)
    

    
    for pos, curr_tweets in [(1, tweets.pos), (0, tweets.neg)]:
        for tweet in curr_tweets[:tweet_count]:
            
            tokens = tweet_as_tokens(tweet, emb.word_dict)
            dimtokens = len(tokens)   
            if dimtokens == 0 or dimtokens == 1:
                continue
            
            
            z =torch.ones(dimtokens,1)/dimtokens
            
            #x= torch.cat((torch.tensor(emb.ws[tokens,:]),torch.reshape(z,(-1,1))),dim=1)
            x = torch.tensor(emb.ws[tokens,:])
            #print(z)
            #print(z.size())
            #vector with length 201 ( 200 from the glove and 1 having 1/len(tweet)
        
            
            
            
            predictions = torch.mean(model.forward(x), 0, keepdim = True)
            
            test_accs+= accuracy(predictions, pos)/L
            #print(accuracy(predictions, pos)/L)
            
    print(test_accs)
    return test_accs

def test1(emb:Embedding, tweets : Tweets, tweet_count :int, epochs: int, batch_size: int, learning_rate: int, device: str, model, loss_func, optimizer):
    
    losses = np.zeros(epochs // 10)
    test_accs = 0
    epoch_acc = 0
    
    L = 2*tweet_count
    
    print('L')
    print(L)
        
    for pos, curr_tweets in [(1, tweets.pos), (0, tweets.neg)]:
        for tweet in curr_tweets[:tweet_count]:
            
            tokens = tweet_as_tokens(tweet, emb.word_dict)
               
            if len(tokens) == 0 or len(tokens) == 1:
                continue
            
            
            
            z =torch.ones(len(tokens),1)/len(tokens)
            
            x = torch.mul(torch.randn((len(z)),2),z)
            
            #print(z)
            #print(z.size())
            #vector with length 201 ( 200 from the glove and 1 having 1/len(tweet)
        
            
            
            
            predictions = torch.mean(model.forward(x), 0, keepdim = True)
            
            test_accs+= accuracy(predictions, pos)/L
            #print(accuracy(predictions, pos)/L)
            
    print(test_accs)
    return test_accs

class convolutional_nn(torch.nn.Module):

    def __init__(self, n_features, n_filters=10):
        super().__init__()

        self.n_filters = n_filters
        self.conv2 = torch.nn.Conv1d(n_features, n_filters, kernel_size=2)
        self.conv3 = torch.nn.Conv1d(n_features, n_filters, kernel_size=3)
        self.conv4 = torch.nn.Conv1d(n_features, n_filters, kernel_size=4)
        self.conv5 = torch.nn.Conv1d(n_features, n_filters, kernel_size=5)

        temp_size = n_filters
        self.linear1 = torch.nn.Linear(4 * n_filters, temp_size)
        self.linear2 = torch.nn.Linear(temp_size, 2)

    def forward(self, x):
        relu = torch.nn.functional.relu

        # apply filters and take maximum element
        x2 = torch.max(self.conv2(x), dim=2)
        x3 = torch.max(self.conv3(x), dim=2)
        x4 = torch.max(self.conv4(x), dim=2)
        x5 = torch.max(self.conv5(x), dim=2)

        # concatenate
        x = torch.cat((x2.values, x3.values, x4.values, x5.values), dim=1)

        # regularize
        # x = torch.nn.functional.dropout(x, training=self.training)

        # do linear trafo + relu
        x = relu(self.linear1(x))
        x = self.linear2(x)

        return x


class LogisticRegressionModel(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.num_classes = 200 #i changed this for the construct_mean_ws
        self.linear_transform = torch.nn.Linear(self.n_features, self.num_classes)

    def forward(self, x):
        return self.linear_transform(x)



def logistic_regression(x_train, y_train, x_test, y_test, n_features, epochs, learning_rate, batch_size, device):
    loss_func = torch.nn.CrossEntropyLoss()
    model_logreg = LogisticRegressionModel(n_features=n_features)
    optimizer = torch.optim.Adam(model_logreg.parameters(), lr=learning_rate)

    print("Training...")
    train(model_logreg, x_train, y_train, x_test, y_test, loss_func, optimizer, epochs, batch_size, device)


# TODO make including the variance optional

def construct_ws_nn_and_mean_tensors(emb:Embedding, tweets : Tweets, tweet_count :int, epochs: int, batch_size: int, learning_rate: int, device: str):
    assert tweet_count <= len(tweets.pos) and tweet_count <= len(tweets.neg), "Too many tweets"
    
    x = torch.empty(2 * tweet_count, 2 * emb.size)
    y = torch.empty(2 * tweet_count, dtype=torch.long)
    next_i = 0
    
    
    for pos, curr_tweets in [(1, tweets.pos), (0, tweets.neg)]:
        for tweet in curr_tweets[:tweet_count]:
            tokens = tweet_as_tokens(tweet, emb.word_dict)
            if len(tokens) == 0 or len(tokens) == 1:
                continue
            
            xws = torch.tensor(emb.ws[tokens,:])
            #print('xs')
            #print(xws)
            yws = torch.zeros(len(tokens),1)*(1-pos) + torch.ones(len(tokens),1)*pos
            #print(yws.size())
            
            
            
            model_logreg = LogisticRegressionModel(xws.size(1))
            loss_func = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model_logreg.parameters(), lr=learning_rate)  
            
            
            xws_predicted = train1(model_logreg, xws, yws, torch.zeros_like(xws) ,torch.zeros_like(yws), loss_func, optimizer, epochs, batch_size, device)
            #print("xws")
            #print(xws_predicted)
            
           
            
            x[next_i, :emb.size], x[next_i, emb.size:] = torch.var_mean(xws_predicted, dim=0)
            y[next_i] = pos

            next_i = next_i + 1
    # remove excess capacity
    print(f"Dropped {len(x) - next_i} empty tweets")
    x = x[:next_i]
    y = y[:next_i]
    
    print(x.size())
    print(y.size())

    return x, y
    
def construct_ws_nn(emb:Embedding, tweets : Tweets, tweet_count :int, epochs: int, batch_size: int, learning_rate: int, device: str):
    assert tweet_count <= len(tweets.pos) and tweet_count <= len(tweets.neg), "Too many tweets"
    
    #x = torch.tensor(emb.ws)
    #y = torch.empty(tweet_count, dtype=torch.long)
   
    dim = 0
    for pos, curr_tweets in [(1, tweets.pos), (0, tweets.neg)]:
        for tweet in curr_tweets[:tweet_count]:
            tokens = tweet_as_tokens(tweet, emb.word_dict)
            if len(tokens) == 0 or len(tokens) == 1:
                continue
            dim += len(tokens)
    
    
    x = torch.empty(dim, emb.size)
    y = torch.empty(dim, dtype=torch.long)
    z = torch.empty(dim, 1)
    
    next_i = 0
    for pos, curr_tweets in [(1, tweets.pos), (0, tweets.neg)]:
        for tweet in curr_tweets[:tweet_count]:
            tokens = tweet_as_tokens(tweet, emb.word_dict)
            dimtoken = len(tokens)
            
            if len(tokens) == 0 or len(tokens) == 1:
                continue
                
            z[next_i:next_i+dimtoken,:] = 1/dimtoken
            x[next_i:next_i+dimtoken,:emb.size] = torch.tensor(emb.ws[tokens,:])
            #x[next_i:next_i+dimtoken,-1] = 1/dimtoken
            y[next_i:next_i+dimtoken] = pos
            
            next_i += dimtoken
            #w =torch.ones(len(tokens),1)
            #z1= torch.cat((torch.tensor(emb.ws[tokens,:]),torch.reshape(w,(-1,1))/len(tokens)),dim=1)
            #z = torch.cat((z,torch.reshape(w,(-1,1))/len(tokens)),dim = 0)
            #print(z)
            #print(z.size())
            #x = torch.cat((x,z1),dim=0) #vector with length 201 ( 200 from the glove and 1 having 1/len(tweet)
            
            #y = torch.cat((torch.reshape(y,(-1,1)),torch.ones(len(tokens),1,dtype = torch.long)*pos),dim = 0)
            
           
    # remove excess capacity
    #print(f"Dropped {len(x) - next_i} empty tweets")
    #x = x[:next_i]
    #y = y[:next_i]
    
    y = torch.reshape(y,(-1,1))
    print(x.size())
    print(y.size())
    print(z.size())
    
    
    return x, y , z

    
def construct_mean_tensors(emb: Embedding, tweets: Tweets, tweet_count: int):
    assert tweet_count <= len(tweets.pos) and tweet_count <= len(tweets.neg), "Too many tweets"

    x = torch.empty(2 * tweet_count, 2 * emb.size)
    y = torch.empty(2 * tweet_count, dtype=torch.long)
    next_i = 0

    
    for pos, curr_tweets in [(1, tweets.pos), (0, tweets.neg)]:
        for tweet in curr_tweets[:tweet_count]:
            tokens = tweet_as_tokens(tweet, emb.word_dict)
            if len(tokens) == 0 or len(tokens) == 1:
                continue
            
            x[next_i, :emb.size], x[next_i, emb.size:] = torch.var_mean(torch.tensor(emb.ws[tokens, :]), dim=0)
            y[next_i] = pos

            next_i = next_i + 1

    # remove excess capacity
    #print(f"Dropped {len(x) - next_i} empty tweets")
    #x = x[:next_i]
    #y = y[:next_i]
    
    
    print(x.size())
    print(y.size())
    
    return x, y


def construct_sequential_tensors(emb: Embedding, tweets: Tweets, tweet_count: int, max_length: int):
    assert tweet_count <= len(tweets.pos) and tweet_count <= len(tweets.neg), "Too many tweets"

    # todo maybe store index in x instead of expanded embedding
    x = torch.zeros(2 * tweet_count, max_length, emb.size)
    y = torch.empty(2 * tweet_count, dtype=torch.long)
    cropped_count = 0

    tweet_i = 0

    for pos, curr_tweets in [(1, tweets.pos), (0, tweets.neg)]:
        for tweet in curr_tweets[:tweet_count]:
            tokens = tweet_as_tokens(tweet, emb.word_dict)
            if len(tokens) > max_length:
                cropped_count += 1

            for token_i, token in enumerate(tokens):
                if token_i >= max_length:
                    break

                x[tweet_i, token_i, :] = torch.tensor(emb.ws[token, :])

            y[tweet_i] = pos
            tweet_i = tweet_i + 1

    return x, y


def accuracy(y_pred, y) -> float:
    y_pred = torch.argmax(y_pred, dim=1)
    return torch.eq(y_pred, y).float().mean()


def plot_tweet_lengths(tweets: Tweets, max_length: int):
    pos_lengths = list(tweet.count(" ") for tweet in tweets.pos)
    neg_lengths = list(tweet.count(" ") for tweet in tweets.neg)

    pyplot.hist([pos_lengths, neg_lengths], range=(0, max_length), bins=max_length, label=["pos", "neg"], density=True)
    pyplot.legend()
    pyplot.xlabel("word count")
    pyplot.ylabel("frequency")
    pyplot.show()

    dropped_count = sum(1 for length in pos_lengths + neg_lengths if length > max_length)
    total_count = len(pos_lengths) + len(neg_lengths)
    print(f"max_length={max_length} drops {dropped_count} ({dropped_count / total_count:.4f}) tweets")


def main():
    emb = load_embedding("size_200")
    tweets = load_tweets()

    epochs = 60
    learning_rate = 1e-3
    train_ratio = .95
    batch_size = 400
    n_channels = 40
    n_features = emb.size
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device {device}")

    print("Constructing tensors")
    tweet_count = 200_000
    
    tweets_train , tweets_test = Tweets.split(tweets,train_ratio)
    x_train, y_train, z_train = construct_ws_nn(emb, tweets_train, tweet_count , epochs , batch_size, learning_rate, device)
    #x, y = construct_mean_tensors(emb, tweets, tweet_count=200_000)
    #x, y = construct_sequential_tensors(emb=emb, tweets=tweets, tweet_count=10000, max_length=n_channels)

    #x = x.permute(0, 2, 1)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    z_train = z_train.to(device)
    
    print("Split the data")
    #x_train, y_train, z_train, x_test, y_test, z_test = split_data(x, y, z, train_ratio)
    #print(x_train.size())
    #print(y_train.size())
    #print(x_test.size())
    #print(y_test.size())
    loss_func = torch.nn.CrossEntropyLoss(reduction = 'none')

    #model = convolutional_nn(n_features=n_features, n_filters=10)
    model = torch.nn.Sequential(
        torch.nn.Linear(x_train.shape[1], 50),
        torch.nn.ReLU(),
        torch.nn.Dropout(),
        torch.nn.Linear(50, 2),
    )
    #model = torch.nn.Sequential(
    #   torch.nn.Linear(2, 50),
    #    torch.nn.ReLU(),
    #    torch.nn.Dropout(),
    #    torch.nn.Linear(50, 2),
    #)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print("Training...")
    losses, train_accs = train1(model, x_train, y_train, z_train,
                                          loss_func, optimizer, epochs, batch_size, device)
                                          
    epoch_acc = test(emb, tweets_train, tweet_count , epochs , batch_size , learning_rate , device , model, loss_func, optimizer)                                        

    pyplot.plot(losses, label="loss")
    pyplot.plot(train_accs, label="train acc")
    pyplot.plot(test_accs, label="test acc")
    pyplot.legend()
    pyplot.show()


if __name__ == '__main__':
    main()
