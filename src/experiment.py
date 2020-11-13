import numpy as np

with open("../data/output/embedding_words.txt") as f:
    words = np.array([line.strip() for line in f])

embed_w = np.load("../data/output/embedding_w.npy")


def embed(word: str):
    index = np.where(words == word)[0]
    if len(index) == 0:
        raise IndexError(word)
    index = index[0]
    return embed_w[index, :]


def find(w, n: int) -> str:
    dist = np.linalg.norm(embed_w - w, axis=1)
    min_index = np.argsort(dist)[:n]
    return words[min_index]


print("[[ Similar too cool ]]")
for word in find(embed("cool"), 8):
    print(word)

print("[[ Queen? ]]")
for word in find(embed("king") - embed("guy") + embed("girl"), 20):
    print(word)

# TODO: actually try to train a model based on the embeddings
