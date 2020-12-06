import random
from dataclasses import dataclass
from typing import Dict, List, Optional

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
                pos=shuffled_pos[next_i:next_i+count],
                neg=shuffled_neg[next_i:next_i+count],
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


@dataclass
class Embedding:
    words: [str]
    word_dict: Dict[str, int]
    ws: np.ndarray
    size: int

    def embed(self, word: str):
        index = self.word_dict[word]
        return self.ws[index, :]

    def find(self, w: np.ndarray, n: int):
        dist = np.dot(self.ws, w)
        max_index = np.argsort(-dist)[:n]
        return self.words[max_index]


def load_embedding(name: str):
    with open(f"../data/output/emb_words_{name}.txt") as f:
        words = np.array([line.strip() for line in f])
    word_dict = {word: i for i, word in enumerate(words)}

    ws = np.load(f"../data/output/emb_w_{name}.npy")
    ws /= np.linalg.norm(ws, axis=1)[:, np.newaxis]

    return Embedding(words=words, word_dict=word_dict, ws=ws, size=ws.shape[1])


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
