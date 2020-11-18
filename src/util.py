from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class Tweets:
    pos: [str]
    neg: [str]


def load_tweets():
    with open("../data/twitter-datasets/train_pos_full.txt", encoding="latin-1") as f:
        pos = [s.strip() for s in f.readlines()]
    with open("../data/twitter-datasets/train_neg_full.txt", encoding="latin-1") as f:
        neg = [s.strip() for s in f.readlines()]

    return Tweets(pos=pos, neg=neg)


@dataclass
class Embedding:
    words: [str]
    word_dict: Dict[str, int]
    ws: np.ndarray

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

    return Embedding(words=words, word_dict=word_dict, ws=ws)
