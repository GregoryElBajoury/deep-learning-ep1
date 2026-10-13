# TP Réseau de Neurones : Classification binaire sur le dataset Titanic

Ce TP a pour objectif de concevoir, entraîner et évaluer un réseau de neurones multicouches (Sequential API de TensorFlow / Keras) afin de prédire la survie des passagers du Titanic.

## 1. Objectifs Pédagogiques

* Charger et explorer un dataset du monde réel via `seaborn`.
* Mettre en place un pipeline de prétraitement rigoureux (gestion des valeurs manquantes, encodage catégoriel, standardisation) en évitant le *data leakage*.
* Construire, compiler et entraîner un modèle de Deep Learning pour une tâche de **classification binaire**.
* Évaluer les performances du modèle à l'aide de métriques adaptées.

## 2. Exploration et Préparation des Données

### A. Chargement et Sélection des Features

Le jeu de données est chargé directement depuis la bibliothèque `seaborn`. Nous sélectionnons un sous-ensemble de caractéristiques pertinentes (socio-économiques, démographiques et tarifaires) :

* `pclass` : Classe du billet (1ère, 2e, 3e)
* `sex` : Sexe du passager
* `age` : Âge
* `sibsp` : Nombre de frères, sœurs, conjoints à bord
* `parch` : Nombre de parents, enfants à bord
* `fare` : Tarif du billet
* `embarked` : Port d'embarquement (`C`, `Q`, `S`)

La variable cible (`target`) est `survived` ($1$ si le passager a survécu, $0$ sinon).

### B. Division du Dataset (Train / Test Split)

Avant toute modification, nous séparons les données en un jeu d'entraînement ($80\%$) et un jeu de test ($20\%$) en utilisant une stratification (`stratify=y`) pour conserver la même proportion de survivants dans les deux sous-ensembles.

### C. Imputation des Valeurs Manquantes

La colonne `age` contient des valeurs manquantes. Pour éviter toute fuite de données (*data leakage*), les valeurs de remplacement sont calculées **uniquement sur le jeu d'entraînement** :

* L'âge est complété par la médiane du train.
* Le port d'embarquement (`embarked`) est complété par le mode du train.

### D. Encodage et Normalisation

* **Encodage One-Hot (`pd.get_dummies`)** : Transformation des variables textuelles (`sex`, `embarked`, `pclass`) en colonnes binaires, avec `drop_first=True` pour éviter la colinéarité.

* **Standardisation (`StandardScaler`)** : Centrage-réduction des variables continues (`age` et `fare`) basé uniquement sur les paramètres du jeu d'entraînement.

## 3. Architecture du Réseau de Neurones

Le modèle est construit à l'aide de l'API séquentielle de Keras :

```
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(27, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

* **Couche d'entrée** : S'adapte au nombre exact de caractéristiques transformées ($9$ features).
* **Couche cachée** : Composée de $27$ neurones avec une fonction d'activation `ReLU` pour introduire de la non-linéarité.
* **Couche de sortie** : Un unique neurone avec une fonction d'activation `sigmoïde`. Elle renvoie une valeur continue comprise entre $0$ et $1$, interprétée comme la probabilité de survie.

## 4. Entraînement et Évaluation

Le modèle est compilé avec l'optimiseur par défaut (Adam), la fonction de perte `binary_crossentropy` (recommandée pour la classification binaire), et évalué à l'aide de la métrique d'exactitude (`accuracy`).

L'entraînement s'effectue sur plusieurs époques, permettant d'observer la convergence de la fonction de coût et l'amélioration de la précision sur les jeux d'entraînement et de validation.

## 5. Synthèse et Visualisation

À l'instar du notebook IRIS, ce TP met en lumière l'importance cruciale de la phase de préparation des données en Machine Learning / Deep Learning avant de laisser le réseau de neurones apprendre les frontières de décision complexes entre les caractéristiques des passagers et leur survie.

---

## Ressources du TP

Retrouve l'intégralité du code et des expérimentations dans le notebook source sur GitHub :

- 🔗 [Notebook - Réseau de neurones TITANIC (`05_reseau_de_neurones_TITANIC.ipynb`)](https://github.com/GregoryElBajoury/deep-learning-ep1/blob/main/notebooks/5_reseau_neurones_TITANIC.ipynb)

---