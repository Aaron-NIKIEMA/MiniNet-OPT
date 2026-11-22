"""Mises à jour des paramètres par SGD, Momentum et Adam."""

import numpy as np


def initialize_state(name, params):
    """Crée l’état initial nécessaire à l’optimiseur choisi."""
    if name == "sgd":
        return {}
    if name == "momentum":
        return {"velocity": {key: np.zeros_like(value) for key, value in params.items()}}
    if name == "adam":
        zeros = {key: np.zeros_like(value) for key, value in params.items()}
        return {"first_moment": zeros.copy(), "second_moment": zeros.copy(), "step": 0}
    raise ValueError(f"Optimiseur inconnu : {name}")


def sgd_update(params, grads, learning_rate):
    """Effectue une mise à jour SGD simple."""
    return {
        key: params[key] - learning_rate * grads[key]
        for key in params
    }


def momentum_update(params, grads, state, learning_rate, beta=0.9):
    """Effectue une mise à jour Momentum et retourne son nouvel état."""
    velocity = {
        key: beta * state["velocity"][key] + (1 - beta) * grads[key]
        for key in params
    }
    updated_params = {
        key: params[key] - learning_rate * velocity[key]
        for key in params
    }
    return updated_params, {"velocity": velocity}


def adam_update(
    params,
    grads,
    state,
    learning_rate,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
):
    """Effectue une mise à jour Adam avec correction du biais."""
    step = state["step"] + 1
    first_moment = {
        key: beta1 * state["first_moment"][key] + (1 - beta1) * grads[key]
        for key in params
    }
    second_moment = {
        key: beta2 * state["second_moment"][key] + (1 - beta2) * grads[key] ** 2
        for key in params
    }

    updated_params = {}
    for key in params:
        m_hat = first_moment[key] / (1 - beta1**step)
        v_hat = second_moment[key] / (1 - beta2**step)
        updated_params[key] = params[key] - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    new_state = {
        "first_moment": first_moment,
        "second_moment": second_moment,
        "step": step,
    }
    return updated_params, new_state


def update_parameters(name, params, grads, state, learning_rate, **kwargs):
    """Applique une étape d’optimisation via une interface commune."""
    if name == "sgd":
        return sgd_update(params, grads, learning_rate), state
    if name == "momentum":
        return momentum_update(params, grads, state, learning_rate, **kwargs)
    if name == "adam":
        return adam_update(params, grads, state, learning_rate, **kwargs)
    raise ValueError(f"Optimiseur inconnu : {name}")
