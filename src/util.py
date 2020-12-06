import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch


@dataclass
class Tweets:
    pos: [str]
    neg: [str]

    def split(self, counts: [int]) -> ['Tweets']:
        assert sum(counts) <= min(len(self.pos), len(self.neg)), f"Requested too many tweets: {sum(counts)}"

        shuffled_pos = list(self.pos)
        shuffled_neg = list(self.neg)
        random.shuffle(shuffled_pos)
        random.shuffle(shuffled_neg)

        result = []
        next_i = 0

        for count in counts:
            next_i += count

            result.append(Tweets(
                pos=shuffled_pos[next_i:next_i + count],
                neg=shuffled_neg[next_i:next_i + count],
            ))

        return result


def load_tweets():
    with open("../data/twitter-datasets/train_pos_full.txt", encoding="latin-1") as f:
        pos = [s.strip() for s in f.readlines()]
    with open("../data/twitter-datasets/train_neg_full.txt", encoding="latin-1") as f:
        neg = [s.strip() for s in f.readlines()]

    return Tweets(pos=pos, neg=neg)


def tweet_as_tokens(tweet: str, word_dict: dict) -> List[int]:
    """Convert a tweet into a list of word indices"""
    tokens = []
    for word in tweet.split(" "):
        index = word_dict.get(word, None)
        if index is not None:
            tokens.append(index)
    return tokens


def split_tweets(x, y, lens, ratio):
    """
    Split the dataset based according to ratio. 
    """
    perm = np.random.permutation(np.arange(len(x)))
    x, y, lens = x[perm], y[perm], lens[perm]

    split = int(len(x) * ratio)
    return x[:split], y[:split], lens[:split], x[split:], y[split:], lens[split:]


def accuracy(y_pred, y) -> float:
    y_pred = torch.argmax(y_pred, dim=1)
    return torch.eq(y_pred, y).float().mean()


def set_seeds(seed: Optional[int] = None):
    if seed is None:
        seed = random.getrandbits(32)
    print(f"Using seed {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def add_zero_row(ws, DEVICE):
    zeros_row = torch.zeros(ws.shape[1], device=DEVICE)
    ws = torch.cat((zeros_row[None, :], ws), dim=0)
    return ws
