from src.util import load_embedding, load_tweets


def main():
    emb = load_embedding("size_200")
    tweets = load_tweets()


if __name__ == '__main__':
    main()
