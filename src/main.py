import numpy as np
import torch
from matplotlib import pyplot

from util import tweet_as_tokens, Embedding, Tweets, load_embedding, load_tweets, split_data, accuracy


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


def main():
    emb = load_embedding("size_200")
    tweets = load_tweets()

    epochs = 200
    learning_rate = 1e-2
    train_ratio = .95
    batch_size = 5
    n_channels = 40
    n_features = emb.size
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device {device}")

    print("Constructing tensors")
    x, y = construct_sequential_tensors(emb=emb, tweets=tweets, tweet_count=10000, max_length=n_channels)

    x = x.permute(0, 2, 1)
    x = x.to(device)
    y = y.to(device)

    x_train, y_train, x_test, y_test = split_data(x, y, train_ratio)

    loss_func = torch.nn.CrossEntropyLoss()

    model = convolutional_nn(n_features=n_features, n_filters=10)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print("Training...")
    losses, train_accs, test_accs = train(model, x_train, y_train, x_test, y_test,
                                          loss_func, optimizer, epochs, batch_size, device)

    pyplot.plot(losses, label="loss")
    pyplot.plot(train_accs, label="train acc")
    pyplot.plot(test_accs, label="test acc")
    pyplot.legend()
    pyplot.show()


if __name__ == '__main__':
    main()
