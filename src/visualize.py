import numpy as np
import matplotlib.pyplot as plt
import re
from collections import Counter

from split_datasets import load_y, load_tweets_split
from util import Tweets

def plot_tweet_lengths(tweets: Tweets, max_length: int, save=False):
    pos_lengths = list(tweet.count(" ") for tweet in tweets.pos)
    neg_lengths = list(tweet.count(" ") for tweet in tweets.neg)

    plt.hist([neg_lengths, pos_lengths], range=(0, max_length), bins=max_length, label=["negative", "positive"], density=True)
    plt.legend()
    plt.xlabel("Word count")
    plt.ylabel("Frequency")
    if save:
        plt.savefig("tweet_lengths.png", bbox_inches="tight")
    else:
        plt.show()

    dropped_count = sum(1 for length in pos_lengths + neg_lengths if length > max_length)
    total_count = len(pos_lengths) + len(neg_lengths)
    print(f"max_length={max_length} drops {dropped_count} ({dropped_count / total_count:.4f}) tweets")

def word_count(string):
    return(len(string.strip().split(" ")))

def add_word(words, word):
    if word in words:
        words[word] = words[word] + 1
    else:
        words[word] = 1
    return words
    
def plot_word_frequencies(tweets_pos, tweets_neg, save=False):
    words_pos = dict()
    words_neg = dict()

    for i in range(len(tweets_pos)): 
        tweets_pos[i] = re.sub(r'[^\w\s]', '', tweets_pos[i])
        for word in tweets_pos[i].strip().split(" "):
            words_pos = add_word(words_pos, word)

            
    occ_pos = list(words_pos.values())
    occ_pos.sort(reverse=True)
        
        
    for i in range(len(tweets_neg)): 
        tweets_neg[i] = re.sub(r'[^\w\s]', '', tweets_neg[i])
        for word in tweets_neg[i].strip().split(" "):
            words_neg = add_word(words_neg, word)
            
    occ_neg = list(words_neg.values())
    occ_neg.sort(reverse=True)

    plt.plot(occ_neg[1:100], label="negative")
    plt.plot(occ_pos[1:100], label="positive")

    plt.xlabel("Words")
    plt.ylabel("Frequencies")

    plt.legend()
    if save:
        plt.savefig("word_frequencies.png", bbox_inches="tight")
    else:
        plt.show()
"""
tweets_pos = load_y("pos")
tweets_neg = load_y("neg")

plot_word_frequencies(tweets_pos, tweets_neg, save=True)
"""
tweets = load_tweets_split(train_count=1_000_000, test_count=0)
plot_tweet_lengths(tweets[0], max_length=60, save=True)
