from embedding import load_embedding

emb = load_embedding(10_000, 0, True, 200)

# Find words close to the given base word, and print the output as a latex table for the report
print()
BASE_WORDS = ["math", "madrid", "5", "ball", "happy", "sad", "false", "favorite", "!", "wont"]
for word in BASE_WORDS:
    similar_words = emb.find(emb.embed(word), 6)[1:]
    print(f"{word} & {', '.join(similar_words)} \\\\")
print()

# Try to see if the linear relationships work
print("king - man + woman")
print(emb.find(emb.embed("king") - emb.embed("man") + emb.embed("woman"), 10))
print()

# Investigate whether the mean of words corresponds to intuition well
TWEETS = [
    "not happy",
    "not sad",
    "not hungry",
    "very cool",
]

for tweet in TWEETS:
    words = tweet.split(" ")
    mean = sum(emb.embed(w) for w in words) / len(words)
    close = emb.find(mean, 10)
    print(f"mean({words}) -> {close}")
print()
