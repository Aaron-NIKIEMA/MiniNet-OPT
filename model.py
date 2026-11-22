"""Réseau de neurones 2 → 16 → 8 → 1."""

import numpy as np
from data import SEED
from layers import relu, sigmoid
from regularization import dropout_forward


def init_params(seed=SEED):
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0, np.sqrt(2 / 2), size=(16, 2)),
        "b1": np.zeros((1, 16)),
        "W2": rng.normal(0, np.sqrt(2 / 16), size=(8, 16)),
        "b2": np.zeros((1, 8)),
        "W3": rng.normal(0, np.sqrt(1 / 8), size=(1, 8)),
        "b3": np.zeros((1, 1)),
    }


def forward(X, params, training=False, dropout_rate=0.0, rng=None):
    """Calcule les prédictions ; le dropout n’est actif qu’en entraînement."""
    if training and dropout_rate > 0 and rng is None:
        rng = np.random.default_rng()

    Z1 = X @ params["W1"].T + params["b1"]
    A1 = relu(Z1)
    if training:
        A1, mask1 = dropout_forward(A1, dropout_rate, rng)
    else:
        mask1 = np.ones_like(A1)

    Z2 = A1 @ params["W2"].T + params["b2"]
    A2 = relu(Z2)
    if training:
        A2, mask2 = dropout_forward(A2, dropout_rate, rng)
    else:
        mask2 = np.ones_like(A2)

    Z3 = A2 @ params["W3"].T + params["b3"]
    y_hat = sigmoid(Z3)
    cache = (X, Z1, A1, mask1, Z2, A2, mask2, y_hat)
    return y_hat, cache


def backward(y, params, cache):
    """Calcule les gradients de la BCE par rétropropagation."""
    X, Z1, A1, mask1, Z2, A2, mask2, y_hat = cache
    m = y.shape[0]

    # BCE + sigmoid : dérivée simplifiée à la sortie
    dZ3 = (y_hat - y) / m
    dW3 = dZ3.T @ A2
    db3 = np.sum(dZ3, axis=0, keepdims=True)

    dA2 = (dZ3 @ params["W3"]) * mask2
    dZ2 = dA2 * (Z2 > 0)
    dW2 = dZ2.T @ A1
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = (dZ2 @ params["W2"]) * mask1
    dZ1 = dA1 * (Z1 > 0)
    dW1 = dZ1.T @ X
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    return {"W1": dW1, "b1": db1, "W2": dW2,
            "b2": db2, "W3": dW3, "b3": db3}
