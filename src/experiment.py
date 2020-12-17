from embedding import load_embedding

emb = load_embedding(10_000, 0, True, 200)

print()
BASE_WORDS = ["math", "madrid", "5", "ball", "happy", "sad", "false", "favorite", "!", "wont"]
for word in BASE_WORDS:
    similar_words = emb.find(emb.embed(word), 6)[1:]
    print(f"{word} & {', '.join(similar_words)} \\\\")
print()

print("king - man + woman")
print(emb.find(emb.embed("king") - emb.embed("man") + emb.embed("woman"), 10))
print()

print("mean(not happy)")
print(emb.find((emb.embed("not") + emb.embed("happy")) / 2, 20))
print()

print("mean(not sad)")
print(emb.find((emb.embed("not") + emb.embed("sad")) / 2, 20))
print()

print("mean(not hungry)")
print(emb.find((emb.embed("not") + emb.embed("hungry")) / 2, 20))
print()
