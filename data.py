"""Génération, description, découpage et standardisation des données."""

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

SEED = 42


def generate_data(seed=SEED, n_samples=1500, noise=0.20):
    """Génère le jeu de classification binaire make_moons."""
    return make_moons(
        n_samples=n_samples,
        noise=noise,
        random_state=seed,
    )


def describe_data(X, y, name="Données"):
    """Affiche les dimensions et la répartition des classes."""
    print(f"{name} — X : {X.shape}, y : {y.shape}")
    classes, counts = np.unique(y, return_counts=True)
    print(
        "Effectifs par classe :",
        dict(zip(classes.tolist(), counts.tolist())),
    )


def split_data(X, y, seed=SEED):
    """Découpe les données en ensembles stratifiés train/validation/test (60/20/20)."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def standardize_data(X_train, X_val, X_test):
    """Standardise les ensembles avec la moyenne et l’écart-type du train."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    return (
        (X_train - mean) / std,
        (X_val - mean) / std,
        (X_test - mean) / std,
        mean,
        std,
    )


def describe_standardization(mean, std):
    """Affiche les statistiques utilisées pour standardiser les variables."""
    print("Moyenne calculée sur le train :", mean)
    print("Écart-type calculé sur le train :", std)


def describe_standardized_features(X_train):
    """Affiche la moyenne et l’écart-type du train après standardisation."""
    print("Moyenne train après standardisation :", X_train.mean(axis=0))
    print("Écart-type train après standardisation :", X_train.std(axis=0))
