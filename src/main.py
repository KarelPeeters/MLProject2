import numpy as np
import torch
from matplotlib import pyplot
import sys

from util import tweet_as_tokens, Embedding, Tweets, load_embedding, load_tweets, split_data


def train(model, ws, x_train, y_train, lens_train, x_test, y_test, lens_test, loss_func, optimizer, epochs: int, batch_size: int, device: str):
    losses = np.zeros(epochs)
    train_accs = np.zeros(epochs)
    test_accs = np.zeros(epochs)

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
            lens_batch = lens_train[batch_i]

            predictions = model.forward(x_train_batch, lens_batch, ws)

            loss = loss_func(predictions, y_train_batch)

            epoch_loss += loss.item() / batch_count
            epoch_acc += accuracy(predictions, y_train_batch) / batch_count

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        y_test_pred = model.forward(x_test, lens_test, ws)
        test_acc = accuracy(y_test_pred, y_test)
        print(f'Epoch {epoch}/{epochs}, loss {epoch_loss:.4f} acc {epoch_acc:.4f} test_acc {test_acc:.4f}')

        losses[epoch] = epoch_loss
        train_accs[epoch] = epoch_acc
        test_accs[epoch] = test_acc

    return losses, train_accs, test_accs

class convolutional_nn(torch.nn.Module):

    def __init__(self, n_features, n_convols = 3, n_filters=None, n_filters_const = 10):
        super().__init__()
        
        if n_filters is None:
            self.n_filters = np.ones(n_convols, dtype=int) * n_filters_const
        else:
            self.n_filters = n_filters
            
        self.n_convols = n_convols
        self.convs = torch.nn.ModuleList()
        for i in range(self.n_convols):
            self.convs.append(torch.nn.Conv1d(n_features, self.n_filters[i], kernel_size=i+2, padding = (i+2) // 2))
 
        temp_size = 10
        self.linear1 = torch.nn.Linear(np.sum(self.n_filters), temp_size)
        self.linear2 = torch.nn.Linear(temp_size, 2)
        """
        self.n_filters = n_filters
        self.conv2 = torch.nn.Conv1d(n_features, n_filters_const, kernel_size=2)
        self.conv3 = torch.nn.Conv1d(n_features, n_filters_const, kernel_size=3)
        self.conv4 = torch.nn.Conv1d(n_features, n_filters_const, kernel_size=4)
        self.conv5 = torch.nn.Conv1d(n_features, n_filters_const, kernel_size=5)

        temp_size = n_filters_const
        self.linear1 = torch.nn.Linear(4*n_filters_const, temp_size)
        self.linear2 = torch.nn.Linear(temp_size, 2)
        """
    def forward(self, x, lens, ws):
        x = ws[x, :].permute(0, 2, 1)
        relu = torch.nn.functional.relu
        #apply filters and take maximum element
        max_vals = []
        
        for i in range(self.n_convols):
            conv_size = i + 2
            conv = self.convs[i](x)
            #create a mask that picks out only sensible values of the convolution, i.e. for a tweet of length l and a 
            #convolution with kernel size k, we need to pick the first l - k + 1 values of the convolution 
            arange = torch.arange(x.shape[2] + 1 - conv_size%2, device=x.device)[None, :]
            mask = arange[None, :] < (lens[:, None] + 1 - conv_size%2)
            mask = mask.permute(1, 0, 2)
            
            #replace values that are not sensible with -inf, to exclude them from the maximum 
            neg_inf_tensor = torch.full_like(conv, float("-inf"), device=x.device)
            replaced = torch.where(mask, neg_inf_tensor, conv)

            max_taken = torch.max(replaced, dim=2).values
            #max_taken = torch.max(conv, dim=2).values
            max_vals.append(max_taken)

        #concatenate
        x2 = torch.cat(max_vals, dim=1)
        #x2 = torch.where(x2.eq(float("-inf")), torch.zeros_like(x2), x2)
        x = x2

        #x = torch.max(torch.zeros_like(x), x)

        #regularize
        x = torch.nn.functional.dropout(x, training=self.training)
        """     
        x2 = torch.max(self.convs[0](x), dim=2)
        x3 = torch.max(self.convs[1](x), dim=2)
        x4 = torch.max(self.convs[2](x), dim=2)
        x5 = torch.max(self.convs[3](x), dim=2)
        x = torch.cat((x2.values, x3.values, x4.values, x5.values), dim=1)
        """
        #do linear trafo + relu
        x = relu(self.linear1(x))
        x = self.linear2(x)
        return x

class LogisticRegressionModel(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.num_classes = 2
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
    print(f"Dropped {len(x) - next_i} empty tweets")
    x = x[:next_i]
    y = y[:next_i]

    return x, y


def construct_sequential_tensors(emb: Embedding, tweets: Tweets, tweet_count: int, min_length: int, crop_length: int):
    assert tweet_count <= len(tweets.pos) and tweet_count <= len(tweets.neg), "Too many tweets"

    # todo maybe store index in x instead of expanded embedding
    x = torch.zeros(2 * tweet_count, crop_length, dtype=torch.long)
    y = torch.empty(2 * tweet_count, dtype=torch.long)
    lens = torch.empty(2 * tweet_count, dtype=torch.long)

    cropped_count = 0
    too_short_count = 0
    tweet_i = 0

    for pos, curr_tweets in [(1, tweets.pos), (0, tweets.neg)]:
        for tweet in curr_tweets[:tweet_count]:
            tokens = tweet_as_tokens(tweet, emb.word_dict)
            if len(tokens) < min_length:
                too_short_count += 1
                continue
            if len(tokens) > crop_length:
                cropped_count += 1

            cropped_len = min(crop_length, len(tokens))
            #ATTENTION: +1 for index correction when adding a row of zeros in ws
            x[tweet_i, :cropped_len] = torch.tensor(tokens[:cropped_len]) + 1

            y[tweet_i] = pos
            lens[tweet_i] = cropped_len
            tweet_i = tweet_i + 1

    if cropped_count:
        print(f"Cropped {cropped_count} ({cropped_count / (2 * tweet_count):.4f}) tweets that were too long")
    if too_short_count:
        print(f"Skipped {too_short_count} ({too_short_count / (2 * tweet_count):.4f}) tweets that are too short")

    return x[:tweet_i], y[:tweet_i], lens[:tweet_i]


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
    np.random.seed(123456)
    torch.manual_seed(123456)

    emb = load_embedding("size_200")
    tweets = load_tweets()

    tweet_count = 10
    epochs = 20
    learning_rate = 1e-2
    train_ratio = .95
    batch_size = 5
    n_features = emb.size
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device {device}")

    print("Constructing tensors")
    x, y, lens = construct_sequential_tensors(emb=emb, tweets=tweets, tweet_count=tweet_count, min_length=1, crop_length=40)
    #x, y = construct_mean_tensors(emb=emb, tweets=tweets, tweet_count=1000)

    x = x.to(device)
    y = y.to(device)
    lens = lens.to(device)
    ws = torch.tensor(emb.ws, device=device)
    zeros_row = torch.zeros(ws.shape[1], device=device)
    ws = torch.cat((zeros_row[None, :], ws), dim=0)
    
    x_train, y_train, lens_train, x_test, y_test, lens_test = split_data(x, y, lens, train_ratio)

    loss_func = torch.nn.CrossEntropyLoss()
    model = convolutional_nn(n_features=n_features, n_convols=4, n_filters_const=10)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print("Training...")
    losses, train_accs, test_accs = train(
        model, ws,
        x_train, y_train, lens_train,
        x_test, y_test, lens_test,
        loss_func, optimizer, epochs, batch_size, device
    )

    pyplot.plot(losses, label="loss")
    pyplot.plot(train_accs, label="train acc")
    pyplot.plot(test_accs, label="test acc")
    pyplot.legend()
    pyplot.show()

if __name__ == '__main__':
    main()
