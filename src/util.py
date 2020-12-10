from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import random


@dataclass
class Tweets:
    pos: [str]
    neg: [str]
    
        
    def split(self, train_ratio) -> ('Tweets', 'Tweets'):
        def split_part(tweets):
            shuffled = list(tweets)
            random.shuffle(shuffled)
            split_index = int(train_ratio * len(tweets))
            return tweets[:split_index], tweets[:split_index]

        pos_train, pos_test = split_part(self.pos)
        neg_train, neg_test = split_part(self.neg)
        return Tweets(pos=pos_train, neg=neg_train), Tweets(pos=pos_test, neg=neg_test)


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
    
    
def split_data(x, y, z, ratio):
    """
    Split the dataset based according to ratio. 
    """
    perm = np.random.permutation(np.arange(len(x)))
    x, y, z = x[perm], y[perm], z[perm]

    split = int(len(x) * ratio)
    return x[:split], y[:split], z[:split], x[split:], y[split:], z[split:]

