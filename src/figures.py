import os
from contextlib import redirect_stdout

import numpy as np
from matplotlib import pyplot

from embedding import load_embedding, Embedding
from main import SelectedModel, dispatch_model
from split_datasets import load_tweets_split
from util import TimeEstimator, set_plot_font_size


def main_test_emb_size_gen_data():
    train_count = 100_000
    test_count = 10_000
    epochs = 10
    batch_size = 1000
    model = SelectedModel.MEAN_NEURAL
    k_repeat = 5

    os.makedirs("../figures/emb_size", exist_ok=True)

    emb_sizes = np.array([10, 20, 30, 50, 100, 200])
    # sort in descending order to time estimates are too long, which is better than too short
    emb_sizes[::-1].sort()

    print("Loading embedding")
    full_emb = load_embedding(10_000, 0, True, 200)

    print("Loading tweets")
    tweets_train, tweets_test = load_tweets_split(k_repeat * train_count, k_repeat * test_count)
    tweets_train = tweets_train.split(k_repeat * [train_count])
    tweets_test = tweets_test.split(k_repeat * [test_count])

    print("Start sweep")
    result_test_acc = np.zeros((len(emb_sizes), epochs))
    result_train_acc = np.zeros((len(emb_sizes), epochs))

    timer = TimeEstimator(len(emb_sizes) * k_repeat)

    for i, curr_emb_size in enumerate(emb_sizes):
        print(f"Evaluating embedding size {curr_emb_size}")
        curr_emb = Embedding(full_emb.words, full_emb.word_dict, full_emb.ws[:, :curr_emb_size], curr_emb_size)

        for k in range(k_repeat):
            eta = timer.update(i * k_repeat + k)
            print(f"  k={k}, eta {eta}")

            with redirect_stdout(None):
                _, train_acc, test_acc = \
                    dispatch_model(model, curr_emb, tweets_train[k], tweets_test[k], epochs, batch_size)

            result_train_acc[i, :] += train_acc / k_repeat
            result_test_acc[i, :] += test_acc / k_repeat

    np.save("../figures/emb_size/sizes.npy", emb_sizes)
    np.save("../figures/emb_size/test_acc.npy", result_test_acc)
    np.save("../figures/emb_size/train_acc.npy", result_train_acc)


def main_test_emb_size_plot():
    emb_sizes = np.load("../figures/emb_size/sizes.npy")
    result_test_acc = np.load("../figures/emb_size/test_acc.npy")
    result_train_acc = np.load("../figures/emb_size/train_acc.npy")

    set_plot_font_size()

    pyplot.figure()
    items = pyplot.plot(result_test_acc.T)
    pyplot.gca().set_prop_cycle(None)
    pyplot.plot(result_train_acc.T, '--')
    pyplot.legend(items, [str(s) for s in emb_sizes], title="Embedding size")
    pyplot.xlabel("epoch")
    pyplot.ylabel("accuracy")
    pyplot.savefig("../figures/emb_size/fig_epochs.png")
    pyplot.show()

    pyplot.figure()
    pyplot.plot(emb_sizes, result_test_acc[:, -4:].mean(axis=1), "k", label="Test accuracy")
    pyplot.gca().set_prop_cycle(None)
    pyplot.plot(emb_sizes, result_train_acc[:, -4:].mean(axis=1), "k--", label="Train accuracy")
    pyplot.legend()
    pyplot.xlabel("Embedding size")
    pyplot.ylabel("accuracy")
    pyplot.savefig("../figures/emb_size/fig_size.png")
    pyplot.show()


def main():
    # main_test_emb_size_gen_data()
    main_test_emb_size_plot()


if __name__ == '__main__':
    main()
