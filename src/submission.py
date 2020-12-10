import torch

from embedding import Embedding, load_embedding
from main import construct_sequential_tensors, calc_test_accuracy
from split_datasets import load_tweets_split
from util import DEVICE, set_seeds, Tweets


def save_submission(model, emb: Embedding):
    crop_length = 40

    print("Copying ws")
    ws = torch.tensor(emb.ws, device=DEVICE)

    print("Calculating test accuracy")

    [_, test_tweets] = load_tweets_split(0, 10_000)
    x_test, y_test, lens_test = construct_sequential_tensors(emb, test_tweets, 1, crop_length)
    print("Expected accuracy:", calc_test_accuracy(model, ws, x_test, y_test, lens_test, 1000))

    print("Loading submission tweets")
    with open("../data/test_data.txt") as f:
        submission_tweets = []
        for line in f.readlines():
            tweet = line[line.find(",") + 1:].strip()
            submission_tweets.append(tweet)

    tweet_count = len(submission_tweets)

    # these tweets are not really positive but we ignore y anyway
    tweets = Tweets(pos=submission_tweets, neg=[])

    print("Constructing tensors")
    x_all, _, lens_all = construct_sequential_tensors(emb, tweets, 0, crop_length)

    # ignore the empty tweets
    non_empty_indices, = lens_all.nonzero(as_tuple=True)
    x = x_all[non_empty_indices]
    lens = lens_all[non_empty_indices]

    print("Running through model")
    model.eval()
    y_pred = model.forward(x, lens, ws)
    y_pred_int = y_pred.argmax(dim=1)

    y_pred_int_all = torch.zeros(tweet_count, dtype=torch.long, device=DEVICE)
    y_pred_int_all[non_empty_indices] = y_pred_int

    print("Saving output")
    with open("../data/output/submission.csv", "w") as f:
        f.write("Id,Prediction\n")
        for i in range(tweet_count):
            f.write(f"{i + 1},{y_pred_int_all[i].item() * 2 - 1}\n")


def main():
    print("Loading embedding")
    emb = load_embedding(10_000, 0, 200)

    print("Loading model")
    model = torch.load("../data/output/rnn_98_model.pt")
    model.to(DEVICE)

    save_submission(model, emb)


if __name__ == '__main__':
    set_seeds()
    main()
