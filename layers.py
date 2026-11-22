"""Activations élémentaires du réseau."""

import numpy as np


def relu(z):
    return np.maximum(0, z)


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))
