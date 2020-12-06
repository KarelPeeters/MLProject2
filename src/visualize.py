from matplotlib import pyplot

from util import Tweets


def plot_tweet_lengths(tweets: Tweets, max_length: int):
    pos_lengths = list(tweet.count(" ") for tweet in tweets.pos)
    neg_lengths = list(tweet.count(" ") for tweet in tweets.neg)

    pyplot.hist([pos_lengths, neg_lengths], range=(0, max_length), bins=max_length, label=["pos", "neg"], density=True)
    pyplot.legend()
    pyplot.xlabel("word count")
    pyplot.ylabel("frequency")
    pyplot.show()

    dropped_count = sum(1 for length in pos_lengths + neg_lengths if length > max_length)
    total_count = len(pos_lengths) + len(neg_lengths)
    print(f"max_length={max_length} drops {dropped_count} ({dropped_count / total_count:.4f}) tweets")