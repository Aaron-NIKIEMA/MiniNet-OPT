"""Régularisation L2, dropout et early stopping."""

import numpy as np


def dropout_forward(activations, dropout_rate, rng):
    """Applique le dropout inversé et retourne les activations et le masque."""
    if not 0 <= dropout_rate < 1:
        raise ValueError("dropout_rate doit être compris entre 0 inclus et 1 exclu")
    if dropout_rate == 0:
        return activations, np.ones_like(activations)

    keep_probability = 1 - dropout_rate
    mask = (rng.random(activations.shape) < keep_probability) / keep_probability
    return activations * mask, mask


def parameter_norm(params):
    """Calcule la norme L2 des poids, sans les biais."""
    squared_weights = sum(
        (params[key] ** 2).sum()
        for key in params
        if key.startswith("W")
    )
    return squared_weights**0.5


def l2_penalty(params, coefficient):
    """Calcule λ/2 fois la somme des carrés des poids (sans les biais)."""
    return 0.5 * coefficient * parameter_norm(params) ** 2


def add_l2_gradients(params, grads, coefficient):
    """Ajoute λW aux gradients des poids, sans modifier les gradients d’origine."""
    regularized_grads = {key: value.copy() for key, value in grads.items()}
    for key in params:
        if key.startswith("W"):
            regularized_grads[key] += coefficient * params[key]
    return regularized_grads
