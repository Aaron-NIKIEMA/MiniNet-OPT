"""Visualisation des données et des résultats."""

import matplotlib.pyplot as plt


def plot_data(X, y, title="Jeu de données"):
    """Affiche les points du jeu binaire selon leur classe."""
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=15)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)
    plt.show()


def plot_optimizer_comparison(histories):
    """Trace les métriques d’entraînement des optimiseurs comparés."""
    metrics = (
        ("train_loss", "Loss train"),
        ("val_loss", "Loss validation"),
        ("grad_norm", "Norme du gradient"),
        ("train_accuracy", "Accuracy train"),
        ("val_accuracy", "Accuracy validation"),
    )
    figure, axes = plt.subplots(3, 2, figsize=(12, 12))

    for axis, (key, title) in zip(axes.flat, metrics):
        for name, history in histories.items():
            axis.plot(history[key], label=name)
        axis.set_title(title)
        axis.set_xlabel("Époque")
        axis.grid(True, alpha=0.3)
        axis.legend()

    axes.flat[-1].set_visible(False)
    figure.tight_layout()
    plt.show()


def plot_regularization_comparison(histories):
    """Trace les losses train/validation et la norme des poids par expérience."""
    metrics = (
        ("train_loss", "Loss train"),
        ("val_loss", "Loss validation"),
        ("parameter_norm", "Norme des poids"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))

    for axis, (key, title) in zip(axes, metrics):
        for name, history in histories.items():
            axis.plot(history[key], label=name)
        axis.set_title(title)
        axis.set_xlabel("Époque")
        axis.grid(True, alpha=0.3)
        axis.legend()

    figure.tight_layout()
    plt.show()


def plot_constraint_comparison(histories):
    """Trace les losses, la norme des poids et le résidu de faisabilité."""
    metrics = (
        ("train_loss", "Loss train"),
        ("val_loss", "Loss validation"),
        ("parameter_norm", "Norme des poids"),
        ("feasibility_residual", "Résidu de faisabilité"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))

    for axis, (key, title) in zip(axes.flat, metrics):
        for name, history in histories.items():
            axis.plot(history[key], label=name)
        axis.set_title(title)
        axis.set_xlabel("Époque")
        axis.grid(True, alpha=0.3)
        axis.legend()

    figure.tight_layout()
    plt.show()
