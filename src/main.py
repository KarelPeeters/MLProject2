import time
from enum import Enum, auto
from typing import Optional

import numpy as np
import torch
from matplotlib import pyplot

from embedding import Embedding, load_embedding
from util import tweet_as_tokens, Tweets, load_tweets, accuracy, set_seeds

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device {DEVICE}")


def train(
        model, ws,
        x_train, y_train, lens_train,
        x_test, y_test, lens_test,
        loss_func, optimizer, epochs: int, batch_size: int,
        save_model_path: Optional[str] = None,
):
    print("Training")

    def drop_none(*args):
        return [x for x in args if x is not None]

    losses = np.zeros(epochs)
    train_accs = np.zeros(epochs)
    test_accs = np.zeros(epochs)

    for epoch in range(epochs):
        model.train()

        shuffle = torch.randperm(len(x_train), device=DEVICE)
        batch_count = len(x_train) // batch_size

        train_loss = 0
        train_acc = 0

        prev_print_time = time.monotonic()

        for b in range(batch_count):
            batch_i = shuffle[b * batch_size: (b + 1) * batch_size]
            x_train_batch = x_train[batch_i]
            y_train_batch = y_train[batch_i]
            lens_batch = lens_train[batch_i] if lens_test is not None else None

            predictions = model.forward(*drop_none(x_train_batch, lens_batch, ws))

            batch_loss = loss_func(predictions, y_train_batch)
            batch_acc = accuracy(predictions, y_train_batch)

            train_loss += batch_loss.item() / batch_count
            train_acc += batch_acc / batch_count

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            curr_time = time.monotonic()
            if curr_time - prev_print_time > 10:
                print(f"  batch {b}/{batch_count}: loss {batch_loss.item():.4f} train acc {batch_acc:.4f}")
                prev_print_time = curr_time

        # do batching on test as well to conserve memory
        model.eval()
        batch_count = len(x_test) // batch_size
        test_acc = 0

        for b in range(batch_count):
            x_test_batch = x_test[b * batch_size:(b + 1) * batch_size]
            y_test_batch = y_test[b * batch_size:(b + 1) * batch_size]
            lens_batch = lens_test[b * batch_size:(b + 1) * batch_size] if lens_test is not None else None

            predictions = model.forward(*drop_none(x_test_batch, lens_batch, ws))

            batch_acc = accuracy(predictions, y_test_batch)
            test_acc += batch_acc / batch_count

        print(f'Epoch {epoch}/{epochs}, loss {train_loss:.4f} acc {train_acc:.4f} test_acc {test_acc:.4f}')

        losses[epoch] = train_loss
        train_accs[epoch] = train_acc
        test_accs[epoch] = test_acc

        if save_model_path is not None:
            torch.save(model, save_model_path)

    return losses, train_accs, test_accs


class convolutional_nn(torch.nn.Module):
    def __init__(self, n_features, n_convols=3, n_filters=None, n_filters_const=10):
        super().__init__()

        if n_filters is None:
            self.n_filters = np.ones(n_convols, dtype=int) * n_filters_const
        else:
            self.n_filters = n_filters

        self.n_convols = n_convols
        self.convs = torch.nn.ModuleList()
        for i in range(self.n_convols):
            self.convs.append(torch.nn.Conv1d(n_features, self.n_filters[i], kernel_size=i + 2))

        temp_size = 100
        self.linear1 = torch.nn.Linear(int(np.sum(self.n_filters)), temp_size)
        self.linear2 = torch.nn.Linear(temp_size, 2)

    def forward(self, x, lens, ws):
        x = ws[x, :].permute(0, 2, 1)

        relu = torch.nn.functional.relu
        # apply filters and take maximum element
        max_vals = []
        # print("x=", x.shape)
        for i in range(self.n_convols):
            conv_size = i + 2
            conv = self.convs[i](x)

            arange = torch.arange(x.shape[2] - conv_size + 1, device=x.device)[None, :]
            mask = arange[None, :] < (lens[:, None] - conv_size + 1)
            mask = mask.permute(1, 0, 2)

            neg_inf_tensor = torch.full_like(conv, float("-inf"), device=x.device)
            replaced = torch.where(mask, neg_inf_tensor, conv)

            max_taken = torch.max(replaced, dim=2).values
            max_vals.append(max_taken)

        # concatenate
        x2 = torch.cat(max_vals, dim=1)
        x2 = torch.where(x2.eq(float("-inf")), torch.zeros_like(x2), x2)
        x = x2

        x = torch.max(torch.zeros_like(x), x)

        # regularize
        x = torch.nn.functional.dropout(x, training=self.training)

        # do linear transform + relu
        x = relu(self.linear1(x))
        x = self.linear2(x)
        return x


# TODO make including the variance optional
def construct_mean_tensors(emb: Embedding, tweets: Tweets, include_var: bool):
    total_tweet_count = len(tweets.pos) + len(tweets.neg)

    x = torch.empty(total_tweet_count, emb.size * (1 + include_var))
    y = torch.empty(total_tweet_count, dtype=torch.long)
    next_i = 0

    for pos, curr_tweets in [(1, tweets.pos), (0, tweets.neg)]:
        for tweet in curr_tweets:
            tokens = tweet_as_tokens(tweet, emb.word_dict)
            if len(tokens) == 0 or (include_var and len(tokens) == 1):
                continue

            tokens_embedded = torch.tensor(emb.ws[tokens, :])

            if include_var:
                x[next_i, :emb.size], x[next_i, emb.size:] = torch.var_mean(tokens_embedded, dim=0)
            else:
                x[next_i, :emb.size] = torch.mean(tokens_embedded, dim=0)

            y[next_i] = pos
            next_i = next_i + 1

    # remove excess capacity
    print(f"Dropped {len(x) - next_i} tweets that were too short")

    return x[:next_i].to(DEVICE), y[:next_i].to(DEVICE)


def construct_sequential_tensors(emb: Embedding, tweets: Tweets, min_length: int, crop_length: int):
    total_tweet_count = len(tweets.pos) + len(tweets.neg)

    # todo maybe store index in x instead of expanded embedding
    x = torch.zeros(total_tweet_count, crop_length, dtype=torch.long)
    y = torch.empty(total_tweet_count, dtype=torch.long)
    lens = torch.empty(total_tweet_count, dtype=torch.long)

    cropped_count = 0
    too_short_count = 0
    tweet_i = 0

    for pos, curr_tweets in [(1, tweets.pos), (0, tweets.neg)]:
        for tweet in curr_tweets:
            tokens = tweet_as_tokens(tweet, emb.word_dict)
            if len(tokens) < min_length:
                too_short_count += 1
                continue
            if len(tokens) > crop_length:
                cropped_count += 1

            cropped_len = min(crop_length, len(tokens))
            x[tweet_i, :cropped_len] = torch.tensor(tokens[:cropped_len])

            y[tweet_i] = pos
            lens[tweet_i] = cropped_len
            tweet_i = tweet_i + 1

    if cropped_count:
        print(f"Cropped {cropped_count} ({cropped_count / total_tweet_count :.4f}) tweets that were too long")
    if too_short_count:
        print(f"Skipped {too_short_count} ({too_short_count / total_tweet_count :.4f}) tweets that are too short")

    return x[:tweet_i].to(DEVICE), y[:tweet_i].to(DEVICE), lens[:tweet_i].to(DEVICE)


class RecurrentModel(torch.nn.Module):
    def __init__(self, emb_size: int):
        super().__init__()

        HIDDEN_SIZE = 400

        self.lstm = torch.nn.LSTM(
            input_size=emb_size,
            hidden_size=HIDDEN_SIZE, num_layers=3,
        )

        self.seq = torch.nn.Sequential(
            torch.nn.Dropout(),
            torch.nn.Linear(HIDDEN_SIZE, 50),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(50, 2),
        )

    def forward(self, x, lens, ws):
        x = ws[x, :]

        x = torch.nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm.forward(x)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        x = x[torch.arange(len(x)), lens - 1]
        x = self.seq.forward(x)

        return x


def main_mean_neural(emb: Embedding, tweets_train: Tweets, tweets_test: Tweets):
    learning_rate = 1e-3
    include_var = False
    batch_size = 50
    epochs = 20

    print("Constructing tensors")
    x_train, y_train = construct_mean_tensors(emb, tweets_train, include_var)
    x_test, y_test = construct_mean_tensors(emb, tweets_test, include_var)

    print("Building model")
    model = torch.nn.Sequential(
        torch.nn.Linear(x_train.shape[1], 200),
        torch.nn.ReLU(),
        torch.nn.Dropout(),
        torch.nn.Linear(200, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 2),
    ).to(DEVICE)

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return train(
        model, None,
        x_train, y_train, None,
        x_test, y_test, None,
        loss_func, optimizer, epochs, batch_size
    )


def main_cnn(emb: Embedding, tweets_train: Tweets, tweets_test: Tweets):
    learning_rate = 1e-3
    min_length = 5
    crop_length = 40
    epochs = 20
    batch_size = 10

    print("Constructing tensors")
    ws = torch.tensor(emb.ws, device=DEVICE)
    x_train, y_train, lens_train = construct_sequential_tensors(emb, tweets_train, min_length, crop_length)
    x_test, y_test, lens_test = construct_sequential_tensors(emb, tweets_test, min_length, crop_length)

    print("Building model")
    model = convolutional_nn(n_features=emb.size, n_convols=4, n_filters_const=10)
    model.to(DEVICE)

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return train(
        model, ws,
        x_train, y_train, lens_train,
        x_test, y_test, lens_test,
        loss_func, optimizer, epochs, batch_size
    )


def main_rnn(emb: Embedding, tweets_train: Tweets, tweets_test: Tweets):
    learning_rate = 1e-3
    epochs = 40
    min_length = 1
    crop_length = 40

    print("Constructing tensors")
    ws = torch.tensor(emb.ws, device=DEVICE)
    x_train, y_train, lens_train = construct_sequential_tensors(emb, tweets_train, min_length, crop_length)
    x_test, y_test, lens_test = construct_sequential_tensors(emb, tweets_test, min_length, crop_length)

    print("Building model")
    model = RecurrentModel(emb_size=emb.size)
    model.to(DEVICE)

    cost_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return train(
        model, ws,
        x_train, y_train, lens_train, x_test, y_test, lens_test,
        cost_func, optimizer, epochs, batch_size=1000,
        save_model_path="../data/output/rnn_model.pt",
    )


class SelectedModel(Enum):
    CNN = auto()
    RNN = auto()
    MEAN_NEURAL = auto()
    NEURAL_MEAN = auto()


def main_train_model():
    train_count = 400_000
    test_count = 20_000
    selected_model = SelectedModel.RNN

    set_seeds(None)

    print("Loading embedding")
    emb = load_embedding(10_000, 0, 200)

    print("Loading tweets")
    tweets = load_tweets()

    print("Splitting tweets")
    tweets_train, tweets_test = tweets.split([train_count, test_count])

    if selected_model == SelectedModel.CNN:
        losses, train_accs, test_accs = main_cnn(emb, tweets_train, tweets_test)
    elif selected_model == SelectedModel.RNN:
        losses, train_accs, test_accs = main_rnn(emb, tweets_train, tweets_test)
    elif selected_model == SelectedModel.MEAN_NEURAL:
        losses, train_accs, test_accs = main_mean_neural(emb, tweets_train, tweets_test)
    elif selected_model == SelectedModel.NEURAL_MEAN:
        assert False, "Guilherme is implementing this"
    else:
        assert False, f"Unexpected model {selected_model}"

    pyplot.plot(losses, label="loss")
    pyplot.plot(train_accs, label="train acc")
    pyplot.plot(test_accs, label="test acc")
    pyplot.legend()
    pyplot.show()


def save_submission(model, emb: Embedding):
    ws = torch.tensor(emb.ws, device=DEVICE)

    print("Loading submission tweets")
    with open("../data/test_data.txt") as f:
        submission_tweets = []
        for line in f.readlines():
            tweet = line[line.find(",") + 1:]
            submission_tweets.append(tweet)

    tweet_count = len(submission_tweets)

    # these tweets are not really positive but we ignore y anyway
    tweets = Tweets(pos=submission_tweets, neg=[])

    print("Constructing tensors")
    x_all, _, lens_all = construct_sequential_tensors(emb, tweets, 0, 40)

    # ignore the empty tweets
    non_empty_indices, = lens_all.nonzero(as_tuple=True)
    x = x_all[non_empty_indices]
    lens = lens_all[non_empty_indices]

    print("Running through model")
    y_pred = model.forward(x, lens, ws)
    y_pred_int = y_pred.argmax(dim=1)

    y_pred_int_all = torch.zeros(tweet_count, dtype=torch.long, device=DEVICE)
    y_pred_int_all[non_empty_indices] = y_pred_int

    print("Saving output")
    with open("../data/output/submission.csv", "w") as f:
        f.write("Id,Prediction\n")
        for i in range(tweet_count):
            f.write(f"{i + 1},{y_pred_int_all[i].item() * 2 - 1}\n")


def main_submission():
    model = torch.load("../data/output/rnn_98_model.pt")
    emb = load_embedding(10_000, 0, 200)
    save_submission(model, emb)


def main():
    # main_submission()
    main_train_model()


if __name__ == '__main__':
    main()
