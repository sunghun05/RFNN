"""
Train the PyTorch RFNN on (a subset of) MNIST.

Mirror of theano-rfnn/run_rfnn.py:
  - ntrain randomly-chosen training samples (to reproduce Fig. 5 of the paper)
  - batch size 25, Adadelta(rho=0.95, eps=1e-6)
  - learning rate schedule: lr = 5.0 * (epochs - i) / epochs

Usage:
  python run_rfnn.py --epochs 100 --ntrain 60000
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn

from rfnn import RFNN

# NOTE: torchvision is intentionally NOT used. The system Python is 3.11.0rc1,
# whose `sys` module lacks get_int_max_str_digits, which crashes torchvision's
# eager torch._dynamo import. We read the raw IDX ubyte files directly instead
# (same data layout as the original theano-rfnn/mnist_loader.py).

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "mnist")


def _read_images(path, n):
    with open(path, "rb") as fd:
        loaded = np.frombuffer(fd.read(), dtype=np.uint8)
    return loaded[16:].reshape(n, 28 * 28).astype(np.float32) / 255.0


def _read_labels(path, n):
    with open(path, "rb") as fd:
        loaded = np.frombuffer(fd.read(), dtype=np.uint8)
    return loaded[8:].reshape(n).astype(np.int64)


def load_mnist(ntrain):
    trX = _read_images(os.path.join(DATA_DIR, "train-images-idx3-ubyte"), 60000)
    trY = _read_labels(os.path.join(DATA_DIR, "train-labels-idx1-ubyte"), 60000)
    teX = _read_images(os.path.join(DATA_DIR, "t10k-images-idx3-ubyte"), 10000)
    teY = _read_labels(os.path.join(DATA_DIR, "t10k-labels-idx1-ubyte"), 10000)

    trX = torch.from_numpy(trX).view(-1, 1, 28, 28)
    trY = torch.from_numpy(trY)
    teX = torch.from_numpy(teX).view(-1, 1, 28, 28)
    teY = torch.from_numpy(teY)

    perm = torch.randperm(trX.shape[0])
    trX, trY = trX[perm][:ntrain], trY[perm][:ntrain]
    return trX, trY, teX, teY


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", type=int, default=60000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    trX, trY, teX, teY = load_mnist(args.ntrain)
    trX, trY = trX.to(device), trY.to(device)
    teX, teY = teX.to(device), teY.to(device)
    print(f"train: {trX.shape[0]} samples | test: {teX.shape[0]} samples")

    model = RFNN().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params (alphas + final fc): {n_params}")

    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adadelta(model.parameters(), lr=1.0, rho=0.95, eps=1e-6)

    epochs = float(args.epochs)
    bs = args.batch_size
    for i in range(args.epochs):
        lr = 5.0 * (epochs - i) / epochs
        for g in opt.param_groups:
            g["lr"] = lr

        model.train()
        perm = torch.randperm(trX.shape[0], device=device)
        for start in range(0, trX.shape[0] - bs + 1, bs):
            idx = perm[start:start + bs]
            opt.zero_grad()
            loss = criterion(model(trX[idx]), trY[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            preds = []
            for start in range(0, teX.shape[0], 1000):
                preds.append(model(teX[start:start + 1000]).argmax(1))
            acc = (torch.cat(preds) == teY).float().mean().item()
        print(f"epoch {i + 1:3d}/{args.epochs}  lr={lr:.3f}  test_acc={acc:.4f}")

    print(f"Final Accuracy {acc:.6f}")


if __name__ == "__main__":
    main()
