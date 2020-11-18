import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot

from src.util import tweet_as_tokens, Embedding, Tweets, load_embedding, load_tweets


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
    x, y = construct_tensors(tweet_count=100_000)


if __name__ == '__main__':
    main()
