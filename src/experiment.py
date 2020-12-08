import numpy as np

from embedding import load_embedding

emb = load_embedding(10_000, 3, 200)

print("[[ Similar to cool ]]")
for word in emb.find(emb.embed("cool"), 20):
    print(word)

print("[[ Queen? ]]")
for word in emb.find(emb.embed("king") - emb.embed("guy") + emb.embed("girl"), 8):
    print(word)

