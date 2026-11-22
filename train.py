"""Boucles d’entraînement et calcul des métriques."""

import numpy as np

from data import SEED
from constraints import projected_gradient_step
from losses import bce
from model import backward, forward
from optimizers import initialize_state, update_parameters
from regularization import add_l2_gradients, l2_penalty, parameter_norm


def train_full_batch(X, y, params, learning_rate=0.1, steps=300):
    """Entraîne par descente de gradient simple sur un batch complet."""
    losses = []

    for _ in range(steps):
        y_hat, cache = forward(X, params)
        losses.append(bce(y_hat, y))
        grads = backward(y, params, cache)

        for key in params:
            params[key] -= learning_rate * grads[key]

    y_hat, _ = forward(X, params)
    losses.append(bce(y_hat, y))
    return params, losses


def evaluate(X, y, params):
    """Calcule la BCE et l’accuracy d’un jeu de données."""
    y_hat, _ = forward(X, params)
    return {
        "loss": bce(y_hat, y),
        "accuracy": np.mean((y_hat >= 0.5) == y),
    }


def gradient_norm(X, y, params, l2_coefficient=0.0):
    """Calcule la norme L2 du gradient sur un jeu de données complet."""
    _, cache = forward(X, params)
    grads = backward(y, params, cache)
    grads = add_l2_gradients(params, grads, l2_coefficient)
    return np.sqrt(sum(np.sum(grad**2) for grad in grads.values()))


def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    params,
    optimizer="sgd",
    learning_rate=0.01,
    epochs=100,
    batch_size=32,
    seed=SEED,
    optimizer_kwargs=None,
    l2_coefficient=0.0,
    dropout_rate=0.0,
    patience=None,
    min_delta=0.0,
    constraint_radius=None,
):
    """Entraîne le réseau et enregistre les métriques à chaque époque."""
    if patience is not None and patience < 1:
        raise ValueError("patience doit être un entier positif")

    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    if optimizer == "projected_gradient" and constraint_radius is None:
        raise ValueError("constraint_radius est requis pour projected_gradient")
    if optimizer != "projected_gradient" and constraint_radius is not None:
        raise ValueError("constraint_radius s’utilise avec projected_gradient")

    rng = np.random.default_rng(seed)
    state = None if optimizer == "projected_gradient" else initialize_state(optimizer, params)
    optimizer_kwargs = optimizer_kwargs or {}
    history = {
        "train_loss": [],
        "train_objective": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "grad_norm": [],
        "parameter_norm": [],
    }
    if constraint_radius is not None:
        history["feasibility_residual"] = []
        history["constraint_active"] = []
    best_val_loss = np.inf
    best_params = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(epochs):
        indices = rng.permutation(len(X_train))

        for start in range(0, len(X_train), batch_size):
            batch_indices = indices[start:start + batch_size]
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]

            _, cache = forward(
                X_batch,
                params,
                training=True,
                dropout_rate=dropout_rate,
                rng=rng,
            )
            grads = backward(y_batch, params, cache)
            grads = add_l2_gradients(params, grads, l2_coefficient)
            if optimizer == "projected_gradient":
                params = projected_gradient_step(
                    params, grads, learning_rate, constraint_radius
                )
            else:
                params, state = update_parameters(
                    optimizer,
                    params,
                    grads,
                    state,
                    learning_rate,
                    **optimizer_kwargs,
                )

        train_metrics = evaluate(X_train, y_train, params)
        val_metrics = evaluate(X_val, y_val, params)
        history["train_loss"].append(train_metrics["loss"])
        history["train_objective"].append(
            train_metrics["loss"] + l2_penalty(params, l2_coefficient)
        )
        history["val_loss"].append(val_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["grad_norm"].append(
            gradient_norm(X_train, y_train, params, l2_coefficient)
        )
        current_norm = parameter_norm(params)
        history["parameter_norm"].append(current_norm)
        if constraint_radius is not None:
            history["feasibility_residual"].append(
                max(0.0, current_norm - constraint_radius)
            )
            history["constraint_active"].append(
                abs(current_norm - constraint_radius) <= 1e-8
            )

        if val_metrics["loss"] < best_val_loss - min_delta:
            best_val_loss = val_metrics["loss"]
            best_params = {key: value.copy() for key, value in params.items()}
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if patience is not None and epochs_without_improvement >= patience:
            break

    if patience is not None:
        params = best_params
    history["best_epoch"] = best_epoch
    history["stopped_epoch"] = len(history["val_loss"])

    return params, history
