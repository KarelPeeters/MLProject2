from typing import List, Callable

import numpy as np
import torch
from torch.optim import Optimizer


def train_embedding(
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
            i_batch = shuffle[b:b + batch_size]

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

    return (wx + wy).detach().cpu().numpy()


def main():
    print("Loading cooc")
    cooc = np.load("../data/output/emb_cooc_full.npy")
    word_count = np.max(cooc[:, 0]) + 1

    print(f"cooc size: {cooc.shape}")
    print(f"word count: {word_count}")

    print("Training embedding")

    def optimizer(params):
        return torch.optim.Adam(params)

    w = train_embedding(
        word_count=word_count, cooc=cooc, size=400,
        epochs=5, batch_size=200, device="cuda", optimizer=optimizer, n_max=200,
    )
    np.save("../data/output/emb_w_derp.npy", w)


if __name__ == '__main__':
    main()
