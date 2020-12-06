import time
from enum import Enum, auto

import numpy as np
import torch
from matplotlib import pyplot

from embedding import Embedding, load_embedding
from util import tweet_as_tokens, Tweets, load_tweets, accuracy, set_seeds, add_zero_row

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device {DEVICE}")

def train(
        model, ws, 
        x_train, y_train, lens_train, 
        x_test, y_test, lens_test, loss_func, optimizer, epochs: int, batch_size: int, 
        print_update=True, convolutional=False
):
    def drop_none(*args):
        return [x for x in args if x is not None]
    if print_update:
        print("Training")
    losses = np.zeros(epochs)
    train_accs = np.zeros(epochs)
    test_accs = np.zeros(epochs)

    for epoch in range(epochs):
        model.train()

        shuffle = torch.randperm(len(x_train), device=DEVICE)
        batch_count = len(x_train) // batch_size

        epoch_loss = 0
        epoch_acc = 0

        prev_print_time = time.monotonic()

        for b in range(batch_count):
            batch_i = shuffle[b * batch_size: (b + 1) * batch_size]
            x_train_batch = x_train[batch_i]
            y_train_batch = y_train[batch_i]
            lens_batch = lens_train[batch_i] if lens_test is not None else None

            predictions = model.forward(*drop_none(x_train_batch, lens_batch, ws))

            batch_loss = loss_func(predictions, y_train_batch)
            batch_acc = accuracy(predictions, y_train_batch)

            epoch_loss += batch_loss.item() / batch_count
            epoch_acc += batch_acc / batch_count

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            if print_update:
                curr_time = time.monotonic()
                if curr_time - prev_print_time > 10:
                    print(f"  batch {b}/{batch_count}: loss {batch_loss.item():.2f} train acc {batch_acc:.2f}")
                    prev_print_time = curr_time

        if convolutional:
            y_test_pred = model.forward(*drop_none(x_test, lens_test, ws), training=False)
        else:
            y_test_pred = model.forward(*drop_none(x_test, lens_test, ws))
            
        test_acc = accuracy(y_test_pred, y_test)
        if print_update:
            print(f'Epoch {epoch}/{epochs}, loss {epoch_loss:.4f} acc {epoch_acc:.4f} test_acc {test_acc:.4f}')

        losses[epoch] = epoch_loss
        train_accs[epoch] = epoch_acc
        test_accs[epoch] = test_acc

    return losses, train_accs, test_accs

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
            #ATTENTION: +1 for index correction when adding a row of zeros in ws
            x[tweet_i, :cropped_len] = torch.tensor(tokens[:cropped_len]) + 1

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

def parameter_scan_cnn(n_features, loss_func, learning_rate, ws, epochs, batch_size, 
                        x_train, y_train, lens_train, x_test, y_test, lens_test):
    #TODO: Find best dropout_rate and best batch_size!
    filters = np.arange(start=5, stop=35, step=10, dtype=int)
    opt_filters = np.zeros(4)
    opt_activation_func = ""
    max_test_acc = float('-inf')
    
    for activation_func in [torch.nn.functional.softmax, torch.nn.functional.relu]: 
        for n_filters_2 in filters:
            for n_filters_3 in filters:
                for n_filters_4 in filters:
                    for n_filters_5 in filters:
                            model = ConvolutionalModule(n_features=n_features, 
                                n_filters=[n_filters_2, n_filters_3, n_filters_4, n_filters_5],
                                activation_func=activation_func, dropout_rate=0.5)
                            model.to(DEVICE)
                            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                            print(f"Training with: \nn_filters_2 = {n_filters_2}, n_filters_3 = {n_filters_3}," +
                                f"n_filters_4 = {n_filters_4}, n_filters_5 = {n_filters_5}" +
                                f"\nactivation_func = {activation_func}")
                            losses, train_accs, test_accs = train(
                                model, ws,
                                x_train, y_train, lens_train,
                                x_test, y_test, lens_test,
                                loss_func, optimizer, epochs, batch_size, print_update=False, convolutional=True
                            )
                            max_test_acc_temp = test_accs.max()
                            print(f"Maximum test accuracy = {max_test_acc_temp}")
                            if max_test_acc_temp > max_test_acc:
                                print("New optimum parameters!")
                                max_test_acc = max_test_acc_temp
                                opt_filters = np.array([n_filters_2, n_filters_3, n_filters_4, n_filters_5])
                                opt_activation_func = activation_func
                                
    return opt_filters, opt_activation_func
    
class ConvolutionalModule(torch.nn.Module):
    def __init__(
            self, 
            n_features, n_convols = 3, n_filters=None, n_filters_const = 10, 
            activation_func=torch.nn.functional.relu, dropout_rate=0.5
    ):
        super().__init__()

        if n_filters is None:
            self.n_filters = np.ones(n_convols, dtype=int) * n_filters_const
        else:
            n_convols = len(n_filters)
            self.n_filters = n_filters

        self.activation_func = activation_func
        self.dropout_rate = dropout_rate
        self.n_convols = n_convols
        self.convs = torch.nn.ModuleList()
        for i in range(self.n_convols):
            self.convs.append(torch.nn.Conv1d(n_features, self.n_filters[i], kernel_size=i+2, padding = (i+2) // 2))
 
        temp_size = 10
        self.linear1 = torch.nn.Linear(np.sum(self.n_filters), temp_size)
        self.linear2 = torch.nn.Linear(temp_size, 2)

    def forward(self, x, lens, ws, training=True):
        x = ws[x, :].permute(0, 2, 1)
        #apply filters and take maximum element
        max_vals = []
        for i in range(self.n_convols):
            conv_size = i + 2
            conv = self.convs[i](x)
            """
            max_taken = torch.max(conv, dim=2).values
            max_vals.append(max_taken)
            """
            #create a mask that picks out only sensible values of the convolution, i.e. for a tweet of length l and a 
            #convolution with kernel size k, we need to pick the first l - k%2 + 1 values of the convolution 
            arange = torch.arange(x.shape[2] + 1 - conv_size%2, device=DEVICE)[None, :]
            mask = arange[None, :] < (lens[:, None] + 1 - conv_size%2)
            mask = mask.permute(1, 0, 2)
            
            #replace values that are not sensible with -inf, to exclude them from the maximum 
            neg_inf_tensor = torch.full_like(conv, float("-inf"), device=DEVICE)
            replaced = torch.where(mask, conv, neg_inf_tensor)
            max_taken = torch.max(replaced, dim=2).values
            max_vals.append(max_taken)

        #concatenate
        x = torch.cat(max_vals, dim=1)

        #regularize
        if training:
            x = torch.nn.functional.dropout(x, p=self.dropout_rate, training=self.training)

        #do linear trafo + softmax
        if self.activation_func == torch.nn.functional.softmax:
            x = self.activation_func(self.linear1(x), dim=1)
        else:
            x = self.activation_func(self.linear1(x))
            
        x = self.linear2(x)
        return x

    
def main_mean_neural(emb: Embedding, tweets_train: Tweets, tweets_test: Tweets):
    learning_rate = 1e-3
    include_var = False
    batch_size = 50
    epochs = 100

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


def main_rnn(emb: Embedding, tweets_train: Tweets, tweets_test: Tweets):
    learning_rate = 1e-3
    epochs = 20
    min_length = 1
    crop_length = 40

    print("Constructing tensors")
    ws = torch.tensor(emb.ws, device=DEVICE)
    ws = add_zero_row(ws, DEVICE=DEVICE)
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
        loss_func=cost_func, optimizer=optimizer, epochs=epochs, batch_size=10,
    )

def main_cnn(emb: Embedding, tweets_train: Tweets, tweets_test: Tweets):
    learning_rate = 1e-3
    min_length = 5
    crop_length = 40
    epochs = 10
    n_features = emb.size
    batch_size = 10
    n_filters = [10, 10, 10, 10]
    activation_func = torch.nn.functional.relu

    print("Constructing tensors")
    ws = torch.tensor(emb.ws, device=DEVICE)
    ws = add_zero_row(ws, DEVICE=DEVICE)
    x_train, y_train, lens_train = construct_sequential_tensors(emb, tweets_train, min_length, crop_length)
    x_test, y_test, lens_test = construct_sequential_tensors(emb, tweets_test, min_length, crop_length)

    loss_func = torch.nn.CrossEntropyLoss()
    
    n_filters, activation_func = parameter_scan_cnn(n_features, loss_func, learning_rate, ws, epochs, batch_size, 
                                                    x_train, y_train, lens_train, x_test, y_test, lens_test)

    print("Building model")
    model = ConvolutionalModule(n_features=n_features, n_filters=n_filters, activation_func=activation_func)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return train(
        model, ws,
        x_train, y_train, lens_train,
        x_test, y_test, lens_test,
        loss_func, optimizer, epochs, batch_size,
        convolutional=True
    )

class SelectedModel(Enum):
    CNN = auto()
    RNN = auto()
    MEAN_NEURAL = auto()
    NEURAL_MEAN = auto()


def main():
    train_count = 100
    test_count = 50
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

if __name__ == '__main__':
    main()
