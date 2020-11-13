import bisect
import os
import time
from collections import Counter
from typing import Optional

import numpy as np
import scipy
import scipy.sparse


def read_tweets(path: str, max_count: Optional[int]) -> [str]:
    """Load up to `max_count` tweets from the given file"""
    result = []

    with open(path, encoding="latin-1") as f:
        for line in f:
            if max_count is not None and len(result) >= max_count:
                break
            result.append(line.strip())

    return result


def select_words(tweets, max_word_count: Optional[int]) -> [str]:
    """Select the `max_word_count` most common words in the given tweets"""
    word_counts = Counter()
    for tweet in tweets:
        for word in tweet.split(" "):
            word_counts[word] += 1

    selection = word_counts.most_common(max_word_count) if max_word_count is not None else word_counts.items()
    min_occurrences = min(pair[1] for pair in selection)
    words = list(pair[0] for pair in selection)

    print(f"Kept {len(words)}/{len(word_counts)} words that occur >= {min_occurrences} times")
    words.sort()
    return words


def tweet_as_tokens(tweet: str, words: [str]):
    """Convert a tweet into a list of word indices"""
    tokens = []
    for word in tweet.split(" "):
        index = bisect.bisect(words, word)
        if index > 0 and words[index - 1] == word:
            tokens.append(index - 1)
    return tokens


def construct_cooc(tweets, words) -> scipy.sparse.coo_matrix:
    """Build a sparse co-occurrence matrix"""
    # TODO it should be possible to make this function a lot faster
    row = []
    col = []
    data = []
    for tweet in tweets:
        tokens = tweet_as_tokens(tweet, words)
        for t0 in tokens:
            for t1 in tokens:
                row.append(t0)
                col.append(t1)
                data.append(1)
    cooc = scipy.sparse.coo_matrix((data, (row, col)))
    cooc.sum_duplicates()
    return cooc


def train_embedding(
        cooc: scipy.sparse.coo_matrix, size: int, epochs: int, batch_size: int,
        eta: float, n_max: int = 100, alpha: float = 3 / 4
):
    """Train a GloVe embedding using batched SGD. `size` is the size of the resulting embedding."""
    # TODO random initialization: should the sigma depend on the size?
    w_x = np.random.normal(size=(cooc.shape[0], size))
    w_y = np.random.normal(size=(cooc.shape[1], size))

    for epoch in range(epochs):
        total_cost = 0

        for i in range(len(cooc.data) // batch_size):
            ix = cooc.row[i:i + batch_size]
            jy = cooc.col[i:i + batch_size]
            n = cooc.data[i:i + batch_size]

            log_n = np.log(n)
            fn = np.minimum(1.0, (n / n_max) ** alpha)

            x, y = w_x[ix, :], w_y[jy, :]
            log_n_pred = np.sum(x * y, axis=1)
            total_cost += np.sum(fn * (log_n - log_n_pred) ** 2)

            # TODO rewrite this using pytorch so we can try different optimizers
            scale = np.sum(2 * eta * fn * (log_n - log_n_pred)) / batch_size
            w_x[ix, :] += scale * y
            w_y[jy, :] += scale * x

        avg_cost = total_cost / (len(cooc.data) // batch_size * batch_size)
        print(f"epoch {epoch}, cost {avg_cost}")

    # TODO try returning w_x + w_y here like in the paper instead
    return w_x


def main():
    MAX_WORD_COUNT = 20_000
    MAX_TWEET_COUNT = 100_000

    os.makedirs("../data/output", exist_ok=True)

    print("Reading tweets")
    tweets_pos = read_tweets("../data/twitter-datasets/train_pos_full.txt", MAX_TWEET_COUNT)
    tweets_neg = read_tweets("../data/twitter-datasets/train_neg_full.txt", MAX_TWEET_COUNT)
    tweets = tweets_pos + tweets_neg

    print("Selecting words")
    words = select_words(tweets, MAX_WORD_COUNT)
    with open("../data/output/embedding_words.txt", mode="w") as f:
        for word in words:
            f.write(word + "\n")

    # TODO maybe save cooc as intermediate result to speed up future training
    print("Constructing cooc matrix")
    cooc = construct_cooc(tweets, words)

    print("Training embedding")
    start = time.perf_counter()
    w = train_embedding(cooc, size=20, epochs=10, batch_size=10, eta=0.001)
    print(f"Training took {time.perf_counter() - start}s")
    np.save("../data/output/embedding_w.npy", w)


if __name__ == '__main__':
    main()
