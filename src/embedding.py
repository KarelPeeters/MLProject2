import subprocess
from dataclasses import dataclass
from typing import List, Callable, Optional, Dict

import numpy as np
import torch
from torch.optim import Optimizer


def get_file_paths(max_word_count: int, context_dist: Optional[int], emb_size: int):
    context_dist = context_dist or 0

    word_file = f"../data/output/emb_words_{max_word_count}.txt"
    cooc_file = f"../data/output/emb_cooc_{max_word_count}_ctx_{context_dist}.npy"
    w_file = f"../data/output/emb_w_{max_word_count}_ctx_{context_dist}_size_{emb_size}.npy"

    return word_file, cooc_file, w_file


def construct_cooc(max_word_count: int, context_dist: Optional[int], input_file: str):
    """Call glove-rs to quickly construct the cooc matrix"""
    context_dist = context_dist or 0

    word_file, cooc_file, _ = get_file_paths(max_word_count, context_dist, 0)

    cargo_args = "cargo run --release --manifest-path=../glove-rs/Cargo.toml --".split(" ")
    glove_rs_args = [
        input_file,
        str(max_word_count), str(context_dist),
        word_file, cooc_file
    ]

    args = cargo_args + glove_rs_args

    print(" ".join(args))
    subprocess.run(args, check=True)


def train_embedding_from_cooc(
        word_count: int, cooc: np.ndarray, size: int, epochs: int, batch_size: int,
        device: str, optimizer: Callable[[List[torch.Tensor]], Optimizer],
        n_max: int = 100, alpha: float = 3 / 4,
):
    """Train a GloVe embedding using batched SGD. `size` is the size of the resulting embedding."""
    wx = torch.normal(0, 1 / size, size=(word_count, size), device=device, requires_grad=True)
    wy = torch.normal(0, 1 / size, size=(word_count, size), device=device, requires_grad=True)

    optimizer = optimizer([wx, wy])

    cooc = cooc.astype(dtype=int)
    ix = torch.tensor(cooc[:, 0], dtype=torch.long, device=device)
    iy = torch.tensor(cooc[:, 1], dtype=torch.long, device=device)
    n = torch.tensor(cooc[:, 2], dtype=torch.float, device=device)

    batch_count = len(cooc.data) // batch_size

    for epoch in range(epochs):
        shuffle = torch.randperm(len(cooc), device=device)

        avg_cost = 0

        for b in range(batch_count):
            i_batch = shuffle[b * batch_size:b * batch_size + batch_size]

            ix_batch = ix[i_batch]
            iy_batch = iy[i_batch]
            n_batch = n[i_batch]

            log_n = torch.log(n_batch)
            fn = torch.min(torch.ones_like(n_batch), (n_batch / n_max) ** alpha)

            x, y = wx[ix_batch, :], wy[iy_batch, :]
            log_n_pred = torch.sum(x * y, dim=1)
            cost = torch.mean(fn * (log_n - log_n_pred) ** 2)

            if cost.item() != cost.item():
                print("Nan!")

            avg_cost += cost.item() / batch_count

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            if b % 1000 == 0:
                print(f"batch {b} cost {cost}")

        print(f"epoch {epoch}, cost {avg_cost}")

    result = (wx + wy).detach().cpu().numpy()
    result /= np.linalg.norm(result, axis=1)[:, np.newaxis]
    return result


def create_embedding(input_file, max_word_count, context_dist, emb_size):
    word_file, cooc_file, w_file = get_file_paths(max_word_count, context_dist, emb_size)

    print("Constructing cooc")
    construct_cooc(max_word_count, context_dist, input_file)

    print("Loading cooc")
    cooc = np.load(cooc_file)
    word_count = np.max(cooc[:, 0]) + 1
    print(f"cooc size: {cooc.shape}")
    print(f"word count: {word_count}")

    print("Training embedding")

    def optimizer(params):
        return torch.optim.Adam(params)

    w = train_embedding_from_cooc(
        word_count=word_count, cooc=cooc, size=200,
        epochs=10, batch_size=200, device="cuda", optimizer=optimizer, n_max=200,
    )

    print("Saving output")
    np.save(w_file, w)


@dataclass
class Embedding:
    words: [str]
    word_dict: Dict[str, int]
    ws: np.ndarray
    size: int

    def embed(self, word: str):
        index = self.word_dict[word]
        return self.ws[index, :]

    def find(self, w: np.ndarray, n: int):
        dist = np.dot(self.ws, w)
        max_index = np.argsort(-dist)[:n]
        return self.words[max_index]


def load_embedding(max_word_count: int, context_dist: Optional[int], emb_size: int):
    word_file, _, w_file = get_file_paths(max_word_count, context_dist, emb_size)

    with open(word_file) as f:
        words = np.array([line.strip() for line in f])
    word_dict = {word: i for i, word in enumerate(words)}

    ws = np.load(w_file)

    return Embedding(words=words, word_dict=word_dict, ws=ws, size=ws.shape[1])


def main():
    # TODO we really need to split the test/train data before this point, right now we're just using all of the data
    input_file = "../data/output/all_tweets.txt"

    max_word_count: int = 10_000
    context_dist: Optional[int] = 3
    emb_size: int = 200

    create_embedding(input_file, max_word_count, context_dist, emb_size)


if __name__ == '__main__':
    main()
