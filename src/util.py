import datetime
import random
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from matplotlib import pyplot

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device {DEVICE}")


@dataclass
class Tweets:
    """A dataset consisting of a list of positive and negative tweets."""
    pos: [str]
    neg: [str]

    def split(self, counts: [int]) -> ['Tweets']:
        """Split this dataset into parts with the given among of tweets of each type, after shuffling."""
        assert sum(counts) <= min(len(self.pos), len(self.neg)), f"Requested too many tweets: {sum(counts)}"

        shuffled_pos = list(self.pos)
        shuffled_neg = list(self.neg)
        random.shuffle(shuffled_pos)
        random.shuffle(shuffled_neg)

        result = []
        next_i = 0

        for count in counts:
            result.append(Tweets(
                pos=shuffled_pos[next_i:next_i + count],
                neg=shuffled_neg[next_i:next_i + count],
            ))

            next_i += count

        return result

    def take(self, count: int):
        """Pick `count` random tweets of each type"""
        return self.split([count])[0]

    def total_length(self):
        """The total number of tweets in this dataset"""
        return len(self.pos) + len(self.neg)


def tweet_as_tokens(tweet: str, word_dict: dict) -> List[int]:
    """Convert a tweet into a list of word indices"""
    tokens = []
    for word in tweet.strip().split(" "):
        index = word_dict.get(word, None)
        if index is not None and word != "(" and word != ")":
            tokens.append(index)
    return tokens


def accuracy(y_pred, y) -> float:
    """Calculate the accuracy of the given predictions"""
    y_pred = torch.argmax(y_pred, dim=1)
    return torch.eq(y_pred, y).float().mean()


def set_seeds(seed: Optional[int] = None):
    """Set and log the random seed of all of the random libraries used."""
    if seed is None:
        seed = random.getrandbits(32)
    print(f"Using seed {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def drop_none(*args):
    """Small utility function to filter out None of the given arguments"""
    return [x for x in args if x is not None]


class TimeEstimator:
    """Small utility class that predicts how long an iterative process will take"""

    def __init__(self, total_progress: float):
        """Create a new TimeEstimator. total_progress represents the amount of work that needs to be done."""
        self._total_progress = total_progress
        self.alpha = 0.9

        self._prev_progress = 0.0
        self._prev_time: Optional[float] = None
        self._ema_total_time = None

    def update(self, progress: float):
        """Update the time left estimation. progress is the amount of work already done on the total_progress scale."""
        now = time.monotonic()

        if self._prev_time is None:
            self._prev_time = now
            self._prev_progress = progress
            return None

        delta_time = now - self._prev_time
        delta_progress = (progress - self._prev_progress) / self._total_progress
        pred_total_time = delta_time / delta_progress

        self._prev_time = now
        self._prev_progress = progress

        if self._ema_total_time is None:
            self._ema_total_time = pred_total_time
        else:
            self._ema_total_time = self.alpha * self._ema_total_time + (1 - self.alpha) * pred_total_time

        time_left = self._ema_total_time * (1 - progress / self._total_progress)
        return str(datetime.timedelta(seconds=time_left)).split(".")[0]


def add_zero_row(ws):
    """Add a zero row at the start of the ws embedding matrix."""
    zeros_row = torch.zeros(1, ws.shape[1], device=ws.device)
    ws = torch.cat((zeros_row, ws), dim=0)
    return ws


def set_plot_font_size():
    """Set larger font sizes for pyplot."""
    params = {
        "legend.fontsize": "large",
        "axes.labelsize": "large",
        "axes.titlesize": "large",
        "xtick.labelsize": "large",
        "ytick.labelsize": "large"
    }
    pyplot.rcParams.update(params)
