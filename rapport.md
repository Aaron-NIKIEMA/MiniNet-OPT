# MiniNet-Opt : conception, entraînement, régularisation et optimisation sous contraintes

**Projet intégrateur — Optimisation pour le Machine Learning**  
**Auteur :** Aaron NIKIEMA  
**Établissement :** Institut 2iE
**Enseignant :** Dr Aïssam JEBRANE 
**Date :** 22/11/2026

## Résumé

Ce projet étudie l’entraînement d’un réseau de neurones entièrement implémenté en NumPy pour une classification binaire. Le réseau comporte deux couches cachées de 16 et 8 neurones, activées par ReLU, et une sortie sigmoïde. La rétropropagation est validée par différences finies avant la comparaison de SGD, Momentum et Adam. La généralisation est ensuite étudiée avec une pénalité L2, le dropout et l’early stopping. Enfin, une projection sur une boule L2 impose une contrainte explicite sur les poids ; cette approche est comparée expérimentalement à la pénalité L2.

Sur le jeu `make_moons`, la meilleure loss de validation parmi les optimiseurs testés est obtenue par Momentum (`0,0825`). L’early stopping sélectionne l’époque 49 et atteint une accuracy de `98,67 %` sur le test. Dans les expériences réalisées, les pénalités L2 fortes et les taux de dropout élevés dégradent les performances. Les méthodes par pénalité et par contrainte produisent des résultats proches lorsque leurs normes de poids finales sont appariées. Les calculs NumPy et PyTorch concordent à la précision machine sur le batch de vérification.

**Mots-clés :** réseau de neurones, rétropropagation, SGD, Momentum, Adam, généralisation, L2, dropout, early stopping, gradient projeté.

## 1. Introduction

L’entraînement d’un réseau de neurones peut être formulé comme la minimisation d’une fonction de perte empirique par rapport à ses paramètres. La qualité d’un modèle ne dépend toutefois pas seulement de sa capacité à réduire la perte d’entraînement. Le choix de l’optimiseur, la régularisation, la sélection des hyperparamètres et la gestion des contraintes influencent aussi sa généralisation.

L’objectif de ce projet est de relier ces notions dans une implémentation progressive. Le réseau, la propagation avant, la rétropropagation et les optimiseurs sont d’abord programmés avec NumPy. La validité des gradients est contrôlée indépendamment. Les optimiseurs et les méthodes de régularisation sont ensuite comparés sur des ensembles d’entraînement et de validation distincts. Une formulation sous contrainte est enfin étudiée à l’aide du gradient projeté. PyTorch sert uniquement de référence de vérification après validation des implémentations NumPy.

Les questions expérimentales principales sont :

- Quel optimiseur réduit le plus rapidement la loss et lequel donne la meilleure validation ?
- Comment L2, le dropout et l’early stopping modifient-ils l’écart entre entraînement et validation ?
- Quel est l’effet d’une borne sur la norme des poids ?
- Les approches par pénalité L2 et par contrainte explicite donnent-elles des résultats comparables à normes similaires ?

## 2. Données et protocole

### 2.1 Jeu de données

Le jeu principal est généré avec `sklearn.datasets.make_moons` : 1 500 observations, un bruit de `0,20` et une graine fixée à `42`. Les deux classes sont équilibrées : 750 observations par classe. Le problème est non linéaire et de faible dimension, avec deux variables d’entrée.

Les données sont séparées de manière stratifiée en trois ensembles : 900 observations pour l’entraînement, 300 pour la validation et 300 pour le test. Les variables sont standardisées avec la moyenne et l’écart-type calculés uniquement sur l’ensemble d’entraînement. Ces mêmes statistiques servent ensuite à transformer la validation et le test. Après standardisation, la moyenne des variables train est numériquement nulle et leur écart-type vaut un.

Le jeu de validation sert à comparer les optimiseurs et à sélectionner les hyperparamètres. Le jeu de test est réservé à l’évaluation finale du modèle sélectionné.

### 2.2 Reproductibilité

La graine `42` est centralisée dans `data.py`. Les expériences comparatives réutilisent le même split, la même initialisation des poids, le même ordre de mini-batchs et un budget d’entraînement identique, sauf indication contraire. Les expériences J2 et J4 emploient des mini-batchs de 32 observations et, pour les comparaisons principales, 100 époques.

Les résultats présentés correspondent principalement à une seule graine. Ils décrivent donc le comportement observé sur ce split et cette initialisation ; ils ne constituent pas une estimation de la variabilité entre répétitions.

## 3. Architecture et fonctions mathématiques

### 3.1 Réseau et propagation avant

L’architecture utilisée est `2 → 16 → 8 → 1`. Pour un batch de (m) observations, (X\in\mathbb{R}^{m\times 2}). Les matrices de poids sont stockées avec la convention `(neurones_sortie, neurones_entrée)`, et les biais avec la forme `(1, neurones_sortie)`.

La propagation avant est :

\[
Z^{(1)}=X(W^{(1)})^T+b^{(1)},\qquad A^{(1)}=\operatorname{ReLU}(Z^{(1)}),
\]
\[
Z^{(2)}=A^{(1)}(W^{(2)})^T+b^{(2)},\qquad A^{(2)}=\operatorname{ReLU}(Z^{(2)}),
\]
\[
Z^{(3)}=A^{(2)}(W^{(3)})^T+b^{(3)},\qquad \hat y=\sigma(Z^{(3)}).
\]

Les poids des couches ReLU sont initialisés avec une échelle de type He ; les biais sont initialisés à zéro. Les intermédiaires du forward sont conservés dans un cache pour la rétropropagation. La ReLU est définie par \(\operatorname{ReLU}(z)=\max(0,z)\), avec dérivée \(1_{z>0}\) pour \(z\ne0\). La sigmoïde est \(\sigma(z)=1/(1+e^{-z})\), et sa dérivée vaut \(\sigma(z)(1-\sigma(z))\).

### 3.2 Fonction de perte

La perte utilisée est l’entropie croisée binaire moyenne :

\[
F(\theta)=-\frac{1}{m}\sum_{i=1}^{m}\left[y_i\log(\hat y_i)+(1-y_i)\log(1-\hat y_i)\right].
\]

Les probabilités sont bornées numériquement dans le calcul de la BCE afin d’éviter les logarithmes de zéro.

### 3.3 Rétropropagation

On note \(\delta^{(l)}=\partial F/\partial Z^{(l)}\). Pour la combinaison sigmoïde et BCE, la dérivée de la loss par rapport au logit de sortie se simplifie :

\[
\delta^{(3)}=\frac{\partial F}{\partial Z^{(3)}}=\frac{\hat y-y}{m}.
\]

Pour les couches cachées, la règle de chaîne donne :

\[
\delta^{(2)}=\delta^{(3)}W^{(3)}\odot 1_{Z^{(2)}>0},\qquad
\delta^{(1)}=\delta^{(2)}W^{(2)}\odot 1_{Z^{(1)}>0}.
\]

Les gradients des paramètres, pour chaque couche \(l\), sont :

\[
\frac{\partial F}{\partial W^{(l)}}=(\delta^{(l)})^T A^{(l-1)},\qquad
\frac{\partial F}{\partial b^{(l)}}=\sum_{i=1}^{m}\delta_i^{(l)},\qquad A^{(0)}=X.
\]

Dans les produits matriciels, les dimensions suivent la convention de stockage donnée plus haut ; le code transpose les matrices selon cette convention. Les sommes sur les exemples donnent un gradient moyen parce que le facteur \(1/m\) est déjà présent dans \(\delta^{(3)}\). À une préactivation exactement nulle, la ReLU n’est pas dérivable ; la convention de l’implémentation est une dérivée nulle. Le gradient checking utilise des paramètres identiques dans les deux calculs et évite les préactivations proches de zéro.

La rétropropagation est vérifiée par différences finies centrées sur les huit premières observations du train. L’erreur relative globale obtenue est `1,79 × 10⁻¹¹` et l’erreur absolue maximale `9,77 × 10⁻¹²`. Sur un batch de 32 observations, 300 mises à jour par descente de gradient font passer la loss de `0,6726` à `0,1666`. Ces contrôles valident les dimensions, les gradients et la capacité du réseau à apprendre sur un petit jeu.

## 4. Comparaison des optimiseurs

### 4.1 Méthodes

SGD met à jour chaque paramètre selon le gradient courant :

\[
\theta_{k+1}=\theta_k-\alpha g_k.
\]

Momentum maintient une moyenne mobile des gradients :

\[
v_{k+1}=\beta v_k+(1-\beta)g_k,\qquad
\theta_{k+1}=\theta_k-\alpha v_{k+1}.
\]

Adam combine une moyenne mobile du gradient et de son carré :

\[
m_k=\beta_1m_{k-1}+(1-\beta_1)g_k,\qquad
v_k=\beta_2v_{k-1}+(1-\beta_2)g_k^2,
\]
\[
\widehat m_k=\frac{m_k}{1-\beta_1^k},\quad
\widehat v_k=\frac{v_k}{1-\beta_2^k},\quad
\theta_{k+1}=\theta_k-\alpha\frac{\widehat m_k}{\sqrt{\widehat v_k}+\varepsilon}.
\]

Les mises à jour sont implémentées élément par élément dans `optimizers.py`.

### 4.2 Vérification d’une mise à jour

Sur le même batch de 32 observations et la même initialisation, la loss initiale vaut `0,67256`. Après une mise à jour, elle vaut `0,65180` avec SGD, `0,65180` avec Momentum et `0,63785` avec Adam. Cette vérification confirme que chacune des trois règles réduit la loss sur ce batch ; elle ne suffit pas, à elle seule, à comparer leur convergence.

### 4.3 Entraînement et sélection du learning rate

Les modèles sont entraînés pendant 100 époques. La boucle enregistre les losses train et validation, les accuracies, la norme du gradient et la norme des poids. Une première comparaison avec SGD `0,1`, Momentum `0,5` et Adam `0,01` donne les résultats suivants :

| Optimiseur | Loss train | Loss validation | Accuracy validation |
|---|---:|---:|---:|
| SGD | 0,0803 | 0,0857 | 0,967 |
| Momentum | 0,0812 | 0,0976 | 0,970 |
| Adam | 0,0770 | 0,0948 | 0,963 |

Une grille de trois learning rates par méthode est ensuite comparée sur validation :

| Optimiseur | Learning rates testés | Meilleur taux (loss validation) | Loss validation | Accuracy validation |
|---|---|---:|---:|---:|
| SGD | 0,01 ; 0,05 ; 0,1 | 0,05 | 0,0840 | 0,970 |
| Momentum | 0,1 ; 0,5 ; 1,0 | 0,1 | 0,0825 | 0,967 |
| Adam | 0,001 ; 0,01 ; 0,03 | 0,01 | 0,0948 | 0,963 |

Avec les taux sélectionnés, Adam fait baisser la loss rapidement au début et atteint une accuracy proche de `0,97` en quelques époques. SGD converge plus progressivement. Momentum obtient la loss validation finale la plus faible dans la grille, de peu devant SGD. Adam présente la loss train la plus faible, mais cela ne se traduit pas ici par la meilleure validation. Les accuracies finales sont proches ; la BCE distingue davantage la qualité des probabilités que la seule accuracy.

Les normes de gradient fluctuent, avec des pics pour les trois méthodes, notamment Momentum et Adam. Ces pics ne sont pas accompagnés d’une hausse durable de la loss validation ni d’une baisse persistante de l’accuracy. Ils indiquent des mises à jour irrégulières, sans divergence visible dans les expériences présentées. Adam atteint le plus rapidement une accuracy proche de sa valeur finale et réduit vite la loss au début ; selon le critère de loss de validation finale, Momentum avec \(\alpha=0,1\) obtient le meilleur résultat de la grille. Les deux critères de « convergence la plus rapide » et de meilleure performance finale ne désignent donc pas nécessairement la même méthode.

**Figure 1 à insérer.** Courbes train/validation, norme du gradient et accuracies avec les learning rates retenus. Exporter le graphique de `J2_optimiseurs.ipynb` après réexécution.

## 5. Régularisation et généralisation

### 5.1 Pénalité L2

La pénalité étudiée est :

\[
F_\lambda(\theta)=F_{train}(\theta)+\frac{\lambda}{2}\sum_\ell\|W^{(\ell)}\|_F^2.
\]

Les biais ne sont pas pénalisés ; le gradient de chaque matrice de poids reçoit le terme additionnel (\lambda W^{(\ell)}). Avec Momentum (`learning_rate=0,1`), les résultats après 100 époques sont :

| \(\lambda\) | Loss train | Loss validation | Accuracy validation | Norme des poids |
|---:|---:|---:|---:|---:|
| 0 | 0,0800 | 0,0825 | 0,967 | 8,389 |
| 0,0001 | 0,0801 | 0,0826 | 0,967 | 8,276 |
| 0,001 | 0,0827 | 0,0855 | 0,963 | 7,382 |
| 0,01 | 0,1178 | 0,1215 | 0,953 | 4,949 |
| 0,1 | 0,3356 | 0,3253 | 0,887 | 1,979 |

La norme des poids diminue avec \(\lambda\), comme attendu. Dans cette grille, L2 n’améliore pas la loss validation par rapport au modèle sans pénalité. Pour `λ=0,01` et `0,1`, les losses train et validation augmentent fortement, ce qui indique une pénalisation excessive et un sous-apprentissage. L’écart absolu entre les deux losses reste réduit dans les expériences ; L2 réduit surtout la norme des poids, sans réduire ici un écart train/validation déjà faible.

### 5.2 Dropout

Le dropout inversé est appliqué aux deux couches cachées en entraînement : les activations conservées sont divisées par leur probabilité de conservation. Il est désactivé lors de la validation et de l’évaluation test. Avec Momentum (`learning_rate=0,1`) :

| Taux de dropout | Loss train | Loss validation | Accuracy validation |
|---:|---:|---:|---:|
| 0 | 0,0800 | 0,0825 | 0,967 |
| 0,2 | 0,0991 | 0,1036 | 0,957 |
| 0,5 | 0,1772 | 0,1777 | 0,893 |

Les losses diminuent pour les trois taux, mais les performances se dégradent à mesure que le dropout augmente. Pour ce réseau de petite taille et ce jeu de données, les taux `0,2` et surtout `0,5` semblent trop forts. La diminution de la norme des poids ne suffit pas à améliorer la validation.

### 5.3 Early stopping

L’early stopping surveille la loss validation avec `patience=15` et `min_delta=10^{-4}`. L’entraînement s’arrête à l’époque 64 et restaure le checkpoint de l’époque 49. Sa loss validation vaut `0,0817` et son accuracy `0,973`. Sans arrêt anticipé, l’entraînement poursuit jusqu’à 300 époques ; son minimum historique de validation apparaît à l’époque 78, mais le modèle retourné à l’époque 300 a une loss validation de `0,0872` et une accuracy de `0,970`. Le début du sur-apprentissage est donc visible après la zone du minimum de validation (autour des époques 49 à 78 dans ces courbes), mais la fluctuation de la validation rend impossible de fixer une frontière exacte à partir d’une seule exécution.

Après sélection sur validation, le modèle avec early stopping est évalué une fois sur le test : BCE `0,0594`, accuracy `0,9867` (296 prédictions correctes sur 300). La loss train continue à diminuer tandis que la validation fluctue après son minimum et que la norme des poids augmente. Cela suggère un début de sur-apprentissage, auquel l’arrêt anticipé répond dans cette expérience.

**Figure 2 à insérer.** Courbes de loss, accuracy et norme des poids pour L2, dropout et early stopping. Exporter les figures de `J3_regularisation.ipynb`.

## 6. Optimisation sous contrainte

### 6.1 Projection et gradient projeté

La contrainte porte uniquement sur les poids :

\[
C_R=\{\theta:\|\theta\|_2\le R\},\qquad
\|\theta\|_2^2=\sum_\ell\|W^{(\ell)}\|_F^2.
\]

Après le pas de gradient, les poids sont projetés sur la boule si leur norme dépasse (R). Les biais restent libres. Le test unitaire part d’une norme `5,9624` et, pour un rayon de 2, vérifie une norme projetée de 2 ; il confirme également que les biais ne changent pas lors de la projection.

### 6.2 Influence du rayon

Le rayon de référence est défini à partir du modèle SGD non contraint (`learning_rate=0,05`, 100 époques) : \(R_0=8,1065\). Les expériences utilisent le même split, la même initialisation et le même budget.

| Rayon | Loss train | Loss validation | Accuracy validation | Norme finale | Contrainte active |
|---:|---:|---:|---:|---:|---|
| 0,5 \(R_0\) | 0,1756 | 0,1786 | 0,923 | 4,0533 | Oui |
| 0,75 \(R_0\) | 0,0979 | 0,1007 | 0,960 | 6,0799 | Oui |
| \(R_0\) | 0,0839 | 0,0840 | 0,970 | 8,1065 | Oui |
| 1,5 \(R_0\) | 0,0839 | 0,0840 | 0,970 | 8,1065 | Non |
| 2 \(R_0\) | 0,0839 | 0,0840 | 0,970 | 8,1065 | Non |

Une borne trop petite réduit la capacité du modèle et augmente la loss validation. À `1,5R₀` et `2R₀`, la contrainte n’est pas active et les résultats rejoignent ceux de la référence non contrainte. Le résidu de faisabilité vaut zéro dans toutes les expériences parce qu’aucun modèle ne dépasse la borne ; ce résidu ne suffit donc pas à déterminer si la contrainte est active.

### 6.3 Pénalité L2 et contrainte explicite

Pour comparer les deux formulations, chaque rayon \(R\) est choisi égal à la norme finale obtenue avec une valeur de \(\lambda\). Les deux modèles utilisent SGD (`learning_rate=0,05`), le même split, la même initialisation et 100 époques.

| \(\lambda\) | Rayon apparié \(R\) | Loss validation L2 | Accuracy L2 | Loss validation contrainte | Accuracy contrainte |
|---:|---:|---:|---:|---:|---:|
| 0,001 | 7,5000 | 0,0865 | 0,970 | 0,0875 | 0,967 |
| 0,01 | 5,0001 | 0,1272 | 0,953 | 0,1264 | 0,953 |
| 0,1 | 1,9704 | 0,3276 | 0,883 | 0,3276 | 0,883 |

Les deux approches donnent ici des losses et accuracies proches lorsque leurs normes finales sont similaires. Leurs trajectoires diffèrent : L2 modifie progressivement le gradient, tandis que la projection maintient les poids sur la frontière de la boule lorsque la contrainte est active. Ces observations sont expérimentales et ne constituent pas une correspondance théorique générale dans un problème non convexe.

**Figure 3 à insérer.** Influence du rayon \(R\) sur les losses, la norme des poids et le résidu de faisabilité ; puis comparaison des trajectoires L2/contrainte à normes appariées.

## 7. Vérification avec PyTorch

Les paramètres NumPy sont copiés dans un réseau PyTorch équivalent à trois couches linéaires avec ReLU, ReLU et sigmoïde. Les deux implémentations utilisent les mêmes poids, biais et batch de 32 observations, en double précision. Sont comparés le forward, la BCE, les gradients et les paramètres après une mise à jour SGD puis Adam.

| Quantité comparée | Écart maximal absolu |
|---|---:|
| Prédictions | \(1,110\times10^{-16}\) |
| BCE | 0 |
| Gradients | \(2,776\times10^{-17}\) |
| Paramètres après SGD | \(5,551\times10^{-17}\) |
| Paramètres après Adam | \(6,939\times10^{-18}\) |

Les écarts sont compatibles avec les arrondis en double précision. Cette vérification soutient la cohérence du forward, du backward et des mises à jour comparées sur ce batch. Elle ne remplace pas une comparaison exhaustive des trajectoires d’entraînement.

## 8. Discussion générale et limites

Les expériences montrent qu’une loss train plus basse n’implique pas nécessairement une meilleure validation. Adam minimise le plus la loss train dans la comparaison retenue, alors que Momentum donne la plus faible loss validation après sélection de son learning rate. Les méthodes de régularisation ne sont pas automatiquement bénéfiques : L2 forte, dropout élevé et contraintes trop serrées dégradent ici les performances, vraisemblablement en limitant excessivement un modèle déjà adapté à un problème de faible dimension.

L’early stopping procure une amélioration modeste sur validation par rapport à la poursuite jusqu’à 300 époques. Le modèle retenu obtient une bonne accuracy test, mais le test ne comporte que 300 observations et n’a été évalué qu’une fois ; l’accuracy mesurée reste donc une estimation ponctuelle.

Les principales limites sont :

1. La plupart des comparaisons reposent sur une seule graine et ne mesurent pas la variabilité entre exécutions.
2. Les grilles de learning rates, de (\lambda), de dropout et de rayons sont limitées ; les meilleurs réglages sont les meilleurs parmi les valeurs testées seulement.
3. `make_moons` est un jeu synthétique de faible dimension ; les résultats ne se généralisent pas directement à des données plus complexes.
4. La comparaison PyTorch vérifie un batch et une mise à jour par optimiseur, pas toutes les trajectoires d’entraînement.
5. Les petites différences de loss entre certains modèles ne sont pas évaluées sur plusieurs seeds ni accompagnées d’intervalles d’incertitude.

### 8.1 Réponses aux questions scientifiques

| Question | Réponse tirée des expériences |
|---|---|
| Quel optimiseur converge le plus rapidement ? | Adam atteint le plus rapidement une accuracy proche de sa valeur finale et fait chuter vite la loss au début. Cette conclusion est qualitative et dépend du critère de convergence retenu. |
| Quel optimiseur a la meilleure validation ? | Momentum avec `learning_rate=0,1` : BCE `0,0825`, accuracy `0,967`, parmi les taux testés. SGD est très proche (`0,0840`, `0,970`). |
| Le meilleur résultat train est-il le meilleur test ? | Adam a la loss train la plus basse dans la comparaison initiale, mais le test final a été réservé au modèle choisi par validation avec early stopping. On ne peut donc pas conclure que le modèle de plus faible loss train aurait eu le meilleur score test ; c’est précisément pourquoi le train seul ne sert pas à choisir le modèle. |
| À partir de quand voit-on du sur-apprentissage ? | La validation atteint son meilleur niveau vers l’époque 49 pour l’expérience avec early stopping, puis le modèle sans arrêt présente un minimum historique à 78 et une validation moins bonne à l’époque 300. Le signal est progressif et fluctuant, pas un seuil net. |
| Que fait L2 à la norme et à l’écart train/validation ? | La norme passe de `8,389` sans pénalité à `1,979` pour `λ=0,1`. L’écart train/validation reste petit dans cette expérience ; les grandes valeurs de λ augmentent les deux losses et sous-apprennent. |
| Quel est l’effet du dropout ? | Les taux `0,2` et `0,5` augmentent les losses et réduisent l’accuracy ; le taux `0,5` est le plus pénalisant. |
| Quel modèle retient l’early stopping ? | Le checkpoint de l’époque 49, avec validation BCE `0,0817` et accuracy `0,973`. Son test donne BCE `0,0594` et accuracy `0,9867`. |
| Quel est l’effet d’une diminution de R ? | Les rayons `0,5R₀` et `0,75R₀` dégradent les losses et l’accuracy. À partir de `R₀`, la référence n’est plus dégradée dans les essais présentés. |
| Quand la contrainte est-elle active ? | Elle est active pour `0,5R₀`, `0,75R₀` et `R₀`, et inactive pour `1,5R₀` et `2R₀`. |
| Quelle différence entre L2 et contrainte ? | L2 ajoute un terme au gradient à chaque mise à jour ; la contrainte projette les poids après le pas. À normes finales appariées, les scores observés sont proches, sans équivalence générale. |
| NumPy et PyTorch concordent-ils ? | Oui : les écarts maximaux sont de l’ordre de `10⁻¹⁶` à `10⁻¹⁸` sur le batch et les mises à jour comparés. |
| Quelles sont les limites ? | Une seule seed pour la plupart des expériences, grilles réduites, données synthétiques et test de 300 exemples ; les résultats quantifient ce protocole particulier, pas une performance moyenne générale. |

## 9. Conclusion

Le projet a permis de construire et vérifier un réseau NumPy, de comparer trois optimiseurs, d’étudier plusieurs mécanismes de régularisation et d’implémenter une méthode de gradient projeté. Le gradient checking et la comparaison PyTorch indiquent que les calculs sont cohérents pour les cas contrôlés. Sur ce split, Momentum avec early stopping fournit le modèle retenu selon la validation ; son accuracy test est `98,67 %`.

Les résultats soulignent surtout le rôle du protocole : les choix sont faits sur validation, les normes sont suivies pour comprendre les contraintes et le test intervient uniquement à la fin. La suite naturelle serait de répéter les expériences sur plusieurs seeds et d’élargir les grilles afin de mieux estimer la stabilité des conclusions.

## Références

1. **Énoncé MiniNet-Opt**, *Projet intégrateur — Optimisation pour le Machine Learning*, document fourni avec le projet.
2. Code du projet : `data.py`, `layers.py`, `model.py`, `losses.py`, `optimizers.py`, `regularization.py`, `constraints.py`, `train.py`, `experiments.py` et `plots.py`.
