import re

import matplotlib.pyplot as plt

from split_datasets import load_tweets_split
from util import Tweets, set_plot_font_size


def plot_tweet_lengths(tweets: Tweets, max_length: int, save=False):
    pos_lengths = list(tweet.count(" ") for tweet in tweets.pos)
    neg_lengths = list(tweet.count(" ") for tweet in tweets.neg)

    plt.hist([neg_lengths, pos_lengths], range=(0, max_length), bins=max_length, label=["negative", "positive"],
             density=True)
    plt.legend()
    plt.xlabel("Tweet length in words")
    plt.ylabel("Frequency")
    plt.savefig("../figures/tweet_lengths.png", bbox_inches="tight")
    plt.show()

    dropped_count = sum(1 for length in pos_lengths + neg_lengths if length > max_length)
    total_count = len(pos_lengths) + len(neg_lengths)
    print(f"max_length={max_length} drops {dropped_count} ({dropped_count / total_count:.4f}) tweets")


def word_count(string):
    return (len(string.strip().split(" ")))


def add_word(words, word):
    if word in words:
        words[word] = words[word] + 1
    else:
        words[word] = 1
    return words


def plot_word_frequencies(tweets: Tweets, save=False):
    words_pos = dict()
    words_neg = dict()

    for i in range(len(tweets.pos)):
        tweets.pos[i] = re.sub(r'[^\w\s]', '', tweets.pos[i])
        for word in tweets.pos[i].strip().split(" "):
            words_pos = add_word(words_pos, word)

    occ_pos = list(words_pos.values())
    occ_pos.sort(reverse=True)

    for i in range(len(tweets.neg)):
        tweets.neg[i] = re.sub(r'[^\w\s]', '', tweets.neg[i])
        for word in tweets.neg[i].strip().split(" "):
            words_neg = add_word(words_neg, word)

    occ_neg = list(words_neg.values())
    occ_neg.sort(reverse=True)

    plt.plot(occ_neg[1:100], label="negative")
    plt.plot(occ_pos[1:100], label="positive")

    plt.xlabel("Words")
    plt.ylabel("Frequencies")

    plt.legend()
    plt.savefig("../figures/word_frequencies.png", bbox_inches="tight")
    plt.show()


def main():
    set_plot_font_size()

    tweets, _ = load_tweets_split(None, 0)

    plot_word_frequencies(tweets, save=True)
    plot_tweet_lengths(tweets, max_length=40, save=True)


if __name__ == '__main__':
    main()
