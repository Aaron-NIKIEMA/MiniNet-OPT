# MiniNet-Opt

Projet d’optimisation d’un réseau de neurones pour la classification binaire. Le réseau et les algorithmes sont implémentés principalement avec NumPy ; les notebooks présentent les étapes et les expériences des cinq jalons.

## Contenu

- `data.py`, `layers.py`, `model.py`, `losses.py` : données, couches, réseau et fonction de perte.
- `optimizers.py`, `regularization.py`, `constraints.py`, `train.py`, `experiments.py`, `plots.py` : entraînement et expériences.
- `J1_reseau_base.ipynb` à `J5_synthese.ipynb` : notebooks des jalons, à exécuter dans l’ordre.
- `rapport.md` et `rapport.tex` : rapport scientifique en Markdown et sa version LaTeX.
- `figures/` : graphiques utilisés dans le rapport LaTeX.

## Installation

Avec Python 3, installer les dépendances :

```bash
python -m pip install numpy scikit-learn matplotlib jupyter
```

Pour exécuter la vérification PyTorch du jalon J5, installer également PyTorch selon l’environnement utilisé :

```bash
python -m pip install torch
```

## Exécution

Depuis le dossier du projet, lancer Jupyter :

```bash
jupyter lab
```

Ouvrir et exécuter les notebooks dans l’ordre J1, J2, J3, J4, puis J5. Les notebooks importent les fonctions des modules `.py` du dossier ; il faut donc démarrer Jupyter depuis ce dossier. La graine commune est `SEED = 42`, définie dans `data.py`.


