import os
import random
from typing import Optional

from util import Tweets

BASE = "../data/split-datasets"
ALL_TRAIN_TWEETS_PATH = BASE + "/train_all.txt"


def load_y(y: str) -> [str]:
    # load all tweets of type y, deduplicate and shuffle them
    with open(f"../data/twitter-datasets/train_{y}_full.txt", encoding="utf-8") as f:
        all_tweets = f.readlines()

    tweets = list(set(all_tweets))
    print(f"Removed {len(all_tweets) - len(tweets)} duplicates of type {y}")

    random.shuffle(tweets)
    return tweets


def create_split_files(force: bool):
    if os.path.exists(BASE) and not force:
        print("Skip creating split files")
        return

    train_count: int = 1_000_000
    test_count: int = 100_000

    print("Creating split files")
    os.makedirs(BASE, exist_ok=True)

    pos = load_y("pos")
    neg = load_y("neg")

    total_count = min(len(pos), len(neg))
    left_count = total_count - train_count - test_count

    print(f"Ended up with {total_count} tweets, split into:")
    print(f"    train {train_count}: {train_count / total_count:.4f}")
    print(f"    test  {test_count}: {test_count / total_count:.4f}")
    print(f"leaving {left_count}: {left_count / total_count:.4f} to have nice numbers")

    print("Saving output files")
    # save all split files
    for (y, y_tweets) in [("pos", pos), ("neg", neg)]:
        tweets_train = y_tweets[:train_count]
        tweets_test = y_tweets[train_count:]

        for part, part_tweets in [("train", tweets_train), ("test", tweets_test)]:
            with open(f"{BASE}/{part}_{y}.txt", "w", encoding="utf-8") as f:
                f.writelines(part_tweets)

    # save common train file for embedding
    with open(ALL_TRAIN_TWEETS_PATH, "w", encoding="utf-8") as f:
        f.writelines(pos[:train_count])
        f.writelines(neg[:train_count])


def load_tweets_split(train_count: Optional[int], test_count: Optional[int]) -> (Tweets, Tweets):
    create_split_files(force=False)

    result = []
    for part, part_count in [("train", train_count), ("test", test_count)]:
        args = dict()

        for y in ["pos", "neg"]:
            tweets = []
            with open(f"{BASE}/{part}_{y}.txt", encoding="utf-8") as f:
                while tweet := f.readline():
                    if part_count is not None and len(tweets) == part_count:
                        break
                    tweets.append(tweet.strip())

            random.shuffle(tweets)
            assert part_count is None or len(tweets) == part_count,\
                f"Not enough tweets, need {part_count} but got {len(tweets)}"
            args[y] = tweets

        result.append(Tweets(**args))

    return tuple(result)


if __name__ == '__main__':
    create_split_files(force=True)
