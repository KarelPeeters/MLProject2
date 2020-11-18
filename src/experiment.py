import numpy as np

from src.util import load_embedding

emb = load_embedding("size_200")

print("[[ Similar to cool ]]")
for word in emb.find(emb.embed("cool"), 20):
    print(word)

print("[[ Queen? ]]")
for word in emb.find(emb.embed("king") - emb.embed("guy") + emb.embed("girl"), 8):
    print(word)

