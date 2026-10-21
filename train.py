import os
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons

from micrograd import Value
from micrograd.nn import MLP

SEED = 1337
N_SAMPLES = 100
STEPS = 100
HIDDEN = [16, 16]
L2 = 1e-4


def scalar(out):
    return out[0] if isinstance(out, list) else out


def loss_fn(model, X, y):
    scores = [scalar(model([Value(x0), Value(x1)])) for x0, x1 in X]
    losses = [(1 + -yi * si).relu() for yi, si in zip(y, scores)]
    data_loss = sum(losses) * (1.0 / len(losses))
    reg_loss = L2 * sum(p * p for p in model.parameters())
    accuracy = np.mean([(yi > 0) == (si.data > 0) for yi, si in zip(y, scores)])
    return data_loss + reg_loss, accuracy


def plot_boundary(model, X, y, path):
    h = 0.25
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = np.array([scalar(model([Value(a), Value(b)])).data > 0 for a, b in grid])
    plt.figure(figsize=(5, 4))
    plt.contourf(xx, yy, Z.reshape(xx.shape), cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Spectral, edgecolors="k")
    plt.title("Decision boundary (micrograd MLP)")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    X, y = make_moons(n_samples=N_SAMPLES, noise=0.1, random_state=SEED)
    y = y * 2 - 1 

    model = MLP(2, HIDDEN + [1])
    print(model)
    print("number of parameters:", len(model.parameters()))

    history = {"loss": [], "acc": []}
    for k in range(STEPS):
        loss, acc = loss_fn(model, X, y)
        model.zero_grad()
        loss.backward()

        lr = 1.0 - 0.9 * k / STEPS  # linear learning-rate decay
        for p in model.parameters():
            p.data -= lr * p.grad

        history["loss"].append(loss.data)
        history["acc"].append(acc)
        if k % 10 == 0 or k == STEPS - 1:
            print(f"step {k:3d}  loss {loss.data:.4f}  accuracy {acc * 100:.1f}%")

    os.makedirs("assets", exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(history["loss"], label="loss")
    ax1.set_xlabel("step")
    ax1.set_ylabel("loss")
    ax2 = ax1.twinx()
    ax2.plot(history["acc"], color="tab:orange", label="accuracy")
    ax2.set_ylabel("accuracy")
    fig.suptitle("Training on make_moons")
    fig.tight_layout()
    fig.savefig("assets/loss_curve.png", dpi=120)
    plt.close(fig)

    plot_boundary(model, X, y, "assets/decision_boundary.png")
    print(f"final accuracy: {history['acc'][-1] * 100:.1f}%")


if __name__ == "__main__":
    main()