import numpy as np
import torch
from matplotlib import pyplot

from util import tweet_as_tokens, Embedding, Tweets, load_embedding, load_tweets, split_data


def train(model, x_train, y_train, x_test, y_test, loss_func, optimizer, epochs: int):
    losses = np.zeros(epochs // 10)
    train_accs = np.zeros(epochs // 10)
    test_accs = np.zeros(epochs // 10)

    for epoch in range(epochs):
        model.train()

        predictions = model.forward(x_train)
        loss = loss_func(predictions, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            acc = accuracy(model, x_train, y_train)
            test_acc = accuracy(model, x_test, y_test)
            print(f'Epoch {epoch}/{epochs}, loss {loss.item():.4f} acc {acc:.4f} test_acc {test_acc:.4f}')

            losses[epoch // 10] = loss.item()
            train_accs[epoch // 10] = acc
            test_accs[epoch // 10] = test_acc

    return losses, train_accs, test_accs


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


def accuracy(model, x, y) -> float:
    model.eval()
    y_pred = model.forward(x)
    y_pred = torch.argmax(y_pred, dim=1)
    return torch.eq(y_pred, y).float().mean()


def main():
    emb = load_embedding("size_200")
    tweets = load_tweets()

    epochs = 200
    learning_rate = 1e-2
    train_ratio = .95

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device {device}")

    print("Constructing tensors")
    x, y = construct_mean_tensors(emb=emb, tweets=tweets, tweet_count=200_000)
    x = x.to(device)
    y = y.to(device)

    x_train, y_train, x_test, y_test = split_data(x, y, train_ratio)

    model = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], 50),
        torch.nn.ReLU(),
        torch.nn.Dropout(),
        torch.nn.Linear(50, 2),
    )
    model.to(device)

    cost_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("Training...")
    losses, train_accs, test_accs = train(model, x_train, y_train, x_test, y_test, cost_func, optimizer, epochs)

    pyplot.plot(losses, label="loss")
    pyplot.plot(train_accs, label="train acc")
    pyplot.plot(test_accs, label="test acc")
    pyplot.legend()
    pyplot.show()


if __name__ == '__main__':
    main()
