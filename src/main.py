import torch

from src.util import load_embedding, load_tweets, tweet_as_tokens


def construct_tensors(tweet_count: int):
    emb = load_embedding("size_200")
    all_tweets = load_tweets()

    assert tweet_count <= len(all_tweets.pos) and tweet_count <= len(all_tweets.neg), "Too many tweets"

    x = torch.empty(2 * tweet_count, emb.size)
    y = torch.empty(2 * tweet_count)

    for j, tweets in enumerate([all_tweets.neg, all_tweets.pos]):
        for i, tweet in enumerate(tweets[:tweet_count]):
            tokens = tweet_as_tokens(tweet, emb.word_dict)
            mean = torch.mean(torch.tensor(emb.ws[tokens, :]), dim=0)

            x[j * tweet_count + i, :] = mean
            y[j * tweet_count + i] = j

    return x, y


def main():
    x, y = construct_tensors(tweet_count=100_000)


if __name__ == '__main__':
    main()
