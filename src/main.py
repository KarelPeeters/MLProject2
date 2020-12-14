import time
from enum import Enum, auto
from typing import Optional

import numpy as np
import torch
from matplotlib import pyplot

from embedding import Embedding, load_embedding
from split_datasets import load_tweets_split
from util import tweet_as_tokens, Tweets, accuracy, set_seeds, drop_none, DEVICE, TimeEstimator, add_zero_row


def calc_test_accuracy(
        model, ws,
        x_test, y_test, lens_test,
        batch_size: int,
):
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

    return test_acc


def train(
        model, ws,
        x_train, y_train, lens_train,
        x_test, y_test, lens_test,
        loss_func, optimizer, epochs: int, batch_size: int,
        save_model_path: Optional[str] = None,
        print_update: bool = True,
):
    if print_update:
        print("Training")

    x_train_test = x_train[:len(x_train)]
    y_train_test = y_train[:len(x_train)]
    lens_train_test = lens_train[:len(x_train)] if lens_train is not None else None

    losses = np.zeros(epochs)
    train_accs = np.zeros(epochs)
    test_accs = np.zeros(epochs)

    batch_count = len(x_train) // batch_size
    timer = TimeEstimator(epochs * batch_count)

    for epoch in range(epochs):
        model.train()

        shuffle = torch.randperm(len(x_train), device=DEVICE)

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

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            curr_time = time.monotonic()
            if print_update and curr_time - prev_print_time > 10:
                eta = timer.update(epoch * batch_size + b)
                print(f"  batch {b}/{batch_count}: loss {batch_loss.item():.4f} train acc {batch_acc:.4f} eta {eta}")
                prev_print_time = curr_time

        train_acc = calc_test_accuracy(model, ws, x_train_test, y_train_test, lens_train_test, batch_size)
        test_acc = calc_test_accuracy(model, ws, x_test, y_test, lens_test, batch_size)
        if print_update:
            eta = timer.update((epoch + 1) * batch_count)
            print(
                f'Epoch {epoch}/{epochs}, loss {train_loss:.4f} acc {train_acc:.4f} test_acc {test_acc:.4f} eta {eta}')

        losses[epoch] = train_loss
        train_accs[epoch] = train_acc
        test_accs[epoch] = test_acc

        if save_model_path is not None:
            torch.save(model, save_model_path)

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

def construct_mean_tensors(emb: Embedding, tweets: Tweets, include_var: bool, zero_row: bool):
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


def construct_ws_nn(emb: Embedding, tweets: Tweets, tweet_count: int, epochs: int, batch_size: int, learning_rate: int,
                    device: str):
    assert tweet_count <= len(tweets.pos) and tweet_count <= len(tweets.neg), "Too many tweets"

    # x = torch.tensor(emb.ws)
    # y = torch.empty(tweet_count, dtype=torch.long)

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

            z[next_i:next_i + dimtoken, :] = 1 / dimtoken
            x[next_i:next_i + dimtoken, :emb.size] = torch.tensor(emb.ws[tokens, :])
            # x[next_i:next_i+dimtoken,-1] = 1/dimtoken
            y[next_i:next_i + dimtoken] = pos

            next_i += dimtoken
            # w =torch.ones(len(tokens),1)
            # z1= torch.cat((torch.tensor(emb.ws[tokens,:]),torch.reshape(w,(-1,1))/len(tokens)),dim=1)
            # z = torch.cat((z,torch.reshape(w,(-1,1))/len(tokens)),dim = 0)
            # print(z)
            # print(z.size())
            # x = torch.cat((x,z1),dim=0) #vector with length 201 ( 200 from the glove and 1 having 1/len(tweet)

            # y = torch.cat((torch.reshape(y,(-1,1)),torch.ones(len(tokens),1,dtype = torch.long)*pos),dim = 0)

    # remove excess capacity
    # print(f"Dropped {len(x) - next_i} empty tweets")
    # x = x[:next_i]
    # y = y[:next_i]

    y = torch.reshape(y, (-1, 1))
    print(x.size())
    print(y.size())
    print(z.size())

    return x, y, z


def construct_sequential_tensors(emb: Embedding, tweets: Tweets, min_length: int, crop_length: int, zero_row: bool):
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

            # +1 for index correction when adding a row of zeros in ws
            x[tweet_i, :cropped_len] = torch.tensor(tokens[:cropped_len]) + int(zero_row)

            y[tweet_i] = pos
            lens[tweet_i] = cropped_len
            tweet_i = tweet_i + 1

    if cropped_count:
        print(f"Cropped {cropped_count} ({cropped_count / total_tweet_count :.4f}) tweets that were too long")
    if too_short_count:
        print(f"Skipped {too_short_count} ({too_short_count / total_tweet_count :.4f}) tweets that are too short")

    return x[:tweet_i].to(DEVICE), y[:tweet_i].to(DEVICE), lens[:tweet_i].to(DEVICE)


def parameter_scan_cnn(n_features, loss_func, learning_rate, ws, epochs, batch_size,
                       x_train, y_train, lens_train, x_test, y_test, lens_test):
    # TODO: Find best dropout_rate and best batch_size!
    filters = np.arange(start=5, stop=35, step=10, dtype=int)
    hidden_sizes = np.array([10, 20, 50, 100])

    np.save("../figures/cnn_sweep/filters.npy", filters)
    np.save("../figures/cnn_sweep/hidden_sizes.npy", hidden_sizes)
    np.save("../figures/cnn_sweep/funcs.npy", ["sigmoid", "relu"])

    opt_filters = np.zeros(4)
    opt_activation_func = ""
    max_test_acc = float('-inf')

    timer = TimeEstimator(2 * len(filters) * len(hidden_sizes))
    time_i = 0

    result_train_acc = np.zeros((2, len(filters), len(hidden_sizes)))
    result_test_acc = np.zeros((2, len(filters), len(hidden_sizes)))

    for i0, activation_func in enumerate([torch.sigmoid, torch.nn.functional.relu]):
        for i1, n_filters in enumerate(filters):
            for i2, hidden_size in enumerate(hidden_sizes):
                print("eta",timer.update(time_i))
                time_i += 1

                model = ConvolutionalModule(n_features=n_features,
                                            n_filters=[n_filters, n_filters, n_filters, n_filters],
                                            activation_func=activation_func, dropout_rate=0.5, hidden_size=hidden_size)
                model.to(DEVICE)
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                print(f"Training with: \nn_filters = {n_filters}, n_filters = {n_filters}," +
                      f"n_filters = {n_filters}, n_filters = {n_filters}" +
                      f"\nactivation_func = {activation_func}")
                losses, train_accs, test_accs = train(
                    model, ws,
                    x_train, y_train, lens_train,
                    x_test, y_test, lens_test,
                    loss_func, optimizer, epochs, batch_size,
                    print_update=False
                )

                max_test_acc_temp = test_accs.max()

                result_train_acc[i0, i1, i2] = train_accs.max()
                result_test_acc[i0, i1, i2] = max_test_acc_temp

                print(f"Maximum test accuracy = {max_test_acc_temp}")
                if max_test_acc_temp > max_test_acc:
                    print("New optimum parameters!")
                    max_test_acc = max_test_acc_temp
                    opt_filters = np.array([n_filters, n_filters, n_filters, n_filters])
                    opt_activation_func = activation_func

    np.save("../figures/cnn_sweep/train_acc.npy", result_train_acc)
    np.save("../figures/cnn_sweep/test_acc.npy", result_test_acc)

    return opt_filters, opt_activation_func


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


class ConvolutionalModule(torch.nn.Module):
    def __init__(
            self,
            n_features, n_convols=3, n_filters=None, n_filters_const=10,
            activation_func=torch.nn.functional.relu, dropout_rate=0.5,
            hidden_size: int = 100,
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
            self.convs.append(torch.nn.Conv1d(n_features, self.n_filters[i], kernel_size=i + 2, padding=(i + 2) // 2))

        self.linear1 = torch.nn.Linear(np.sum(self.n_filters), hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, 2)

    def forward(self, x, lens, ws, training=True):
        x = ws[x, :].permute(0, 2, 1)
        # apply filters and take maximum element
        max_vals = []
        for i in range(self.n_convols):
            conv_size = i + 2
            conv = self.convs[i](x)
            """
            max_taken = torch.max(conv, dim=2).values
            max_vals.append(max_taken)
            """
            # create a mask that picks out only sensible values of the convolution, i.e. for a tweet of length l and a
            # convolution with kernel size k, we need to pick the first l - k%2 + 1 values of the convolution
            arange = torch.arange(x.shape[2] + 1 - conv_size % 2, device=DEVICE)[None, :]
            mask = arange[None, :] < (lens[:, None] + 1 - conv_size % 2)
            mask = mask.permute(1, 0, 2)

            # replace values that are not sensible with -inf, to exclude them from the maximum
            neg_inf_tensor = torch.full_like(conv, float("-inf"), device=DEVICE)
            replaced = torch.where(mask, conv, neg_inf_tensor)
            max_taken = torch.max(replaced, dim=2).values
            max_vals.append(max_taken)

        # concatenate
        x = torch.cat(max_vals, dim=1)

        # regularize
        if training:
            x = torch.nn.functional.dropout(x, p=self.dropout_rate, training=self.training)

        # do linear trafo + softmax
        if self.activation_func == torch.nn.functional.softmax:
            x = self.activation_func(self.linear1(x), dim=1)
        else:
            x = self.activation_func(self.linear1(x))

        x = self.linear2(x)
        return x


def main_mean_neural(emb: Embedding, tweets_train: Tweets, tweets_test: Tweets, epochs: int, batch_size: int):
    learning_rate = 1e-3
    include_var = False

    print("Constructing tensors")
    x_train, y_train = construct_mean_tensors(emb, tweets_train, include_var, zero_row=False)
    x_test, y_test = construct_mean_tensors(emb, tweets_test, include_var, zero_row=False)

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


def main_neural_mean(emb: Embedding, tweets_train: Tweets, tweets_test: Tweets, epochs: int, batch_size: int):
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

    tweets_train, tweets_test = Tweets.split(tweets, train_ratio)
    x_train, y_train, z_train = construct_ws_nn(emb, tweets_train, tweet_count, epochs, batch_size, learning_rate,
                                                device)
    # x, y = construct_mean_tensors(emb, tweets, tweet_count=200_000)
    # x, y = construct_sequential_tensors(emb=emb, tweets=tweets, tweet_count=10000, max_length=n_channels)

    # x = x.permute(0, 2, 1)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    z_train = z_train.to(device)

    print("Split the data")
    # x_train, y_train, z_train, x_test, y_test, z_test = split_data(x, y, z, train_ratio)
    # print(x_train.size())
    # print(y_train.size())
    # print(x_test.size())
    # print(y_test.size())
    loss_func = torch.nn.CrossEntropyLoss(reduction='none')

    # model = convolutional_nn(n_features=n_features, n_filters=10)
    model = torch.nn.Sequential(
        torch.nn.Linear(x_train.shape[1], 50),
        torch.nn.ReLU(),
        torch.nn.Dropout(),
        torch.nn.Linear(50, 2),
    )
    # model = torch.nn.Sequential(
    #   torch.nn.Linear(2, 50),
    #    torch.nn.ReLU(),
    #    torch.nn.Dropout(),
    #    torch.nn.Linear(50, 2),
    # )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print("Training...")
    losses, train_accs = train1(model, x_train, y_train, z_train,
                                loss_func, optimizer, epochs, batch_size, device)

    epoch_acc = test(emb, tweets_train, tweet_count, epochs, batch_size, learning_rate, device, model, loss_func,
                     optimizer)

    pyplot.plot(losses, label="loss")
    pyplot.plot(train_accs, label="train acc")
    pyplot.plot(test_accs, label="test acc")
    pyplot.legend()
    pyplot.show()

def main_rnn(emb: Embedding, tweets_train: Tweets, tweets_test: Tweets, epochs: int, batch_size: int):
    learning_rate = 1e-3
    min_length = 1
    crop_length = 40

    print("Constructing tensors")
    ws = torch.tensor(emb.ws, device=DEVICE)
    x_train, y_train, lens_train = construct_sequential_tensors(emb, tweets_train, min_length, crop_length,
                                                                zero_row=False)
    x_test, y_test, lens_test = construct_sequential_tensors(emb, tweets_test, min_length, crop_length, zero_row=False)

    print("Building model")
    model = RecurrentModel(emb_size=emb.size)
    model.to(DEVICE)

    cost_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return train(
        model, ws,
        x_train, y_train, lens_train, x_test, y_test, lens_test,
        loss_func=cost_func, optimizer=optimizer, epochs=epochs, batch_size=batch_size,
    )


def main_cnn(emb: Embedding, tweets_train: Tweets, tweets_test: Tweets, epochs: int, batch_size: int):
    learning_rate = 1e-3
    min_length = 5
    crop_length = 40
    n_features = emb.size
    n_filters = [40, 40, 40, 40]
    activation_func = torch.nn.functional.relu

    print("Constructing tensors")
    ws = add_zero_row(torch.tensor(emb.ws)).to(DEVICE)
    x_train, y_train, lens_train = construct_sequential_tensors(emb, tweets_train, min_length, crop_length,
                                                                zero_row=True)
    x_test, y_test, lens_test = construct_sequential_tensors(emb, tweets_test, min_length, crop_length,
                                                             zero_row=True)

    loss_func = torch.nn.CrossEntropyLoss()
    n_filters, hidden_size, activation_func = parameter_scan_cnn(n_features, loss_func, learning_rate, ws, epochs,
                                                                 batch_size,
                                                                 x_train, y_train, lens_train, x_test, y_test,
                                                                 lens_test)

    print("Building model")
    model = ConvolutionalModule(n_features=n_features, n_filters=n_filters, activation_func=activation_func,
                                hidden_size=hidden_size)
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return train(
        model, ws,
        x_train, y_train, lens_train,
        x_test, y_test, lens_test,
        loss_func, optimizer, epochs, batch_size,
    )


class SelectedModel(Enum):
    CNN = auto()
    RNN = auto()
    MEAN_NEURAL = auto()
    NEURAL_MEAN = auto()


def dispatch_model(selected_model: SelectedModel, emb: Embedding, tweets_train: Tweets, tweets_test: Tweets,
                   epochs: int, batch_size: int):
    if selected_model == SelectedModel.CNN:
        return main_cnn(emb, tweets_train, tweets_test, epochs, batch_size)
    elif selected_model == SelectedModel.RNN:
        return main_rnn(emb, tweets_train, tweets_test, epochs, batch_size)
    elif selected_model == SelectedModel.MEAN_NEURAL:
        return main_mean_neural(emb, tweets_train, tweets_test, epochs, batch_size)
    elif selected_model == SelectedModel.NEURAL_MEAN:
        assert False, "Guilherme is implementing this"
    else:
        assert False, f"Unexpected model {selected_model}"


def main():
    train_count = 100_000
    test_count = 10_000
    selected_model = SelectedModel.CNN
    epochs = 10
    batch_size = 1000

    print("Loading embedding")
    emb = load_embedding(10_000, 0, 200)

    print("Loading tweets")
    tweets_train, tweets_test = load_tweets_split(train_count, test_count)

    print("Training model")
    losses, train_accs, test_accs = dispatch_model(selected_model, emb, tweets_train, tweets_test, epochs, batch_size)

    print("Generating final plot")
    pyplot.plot(losses, label="loss")
    pyplot.plot(train_accs, label="train acc")
    pyplot.plot(test_accs, label="test acc")
    pyplot.ylabel("epoch")
    pyplot.xlabel("performance")
    pyplot.legend()
    pyplot.show()


if __name__ == '__main__':
    set_seeds()
    main()
