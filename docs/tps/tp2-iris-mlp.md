# TP2 : Classification des fleurs Iris avec un MLP

## Objectif
Classifier des fleurs Iris en 3 catégories à partir de 4 mesures morphologiques (longueur/largeur des sépales et des pétales) en utilisant un réseau de neurones multicouches (MLP) avec **Keras**.

---

## 1. Étapes de préparation des données
1. **Séparation Train/Test** : Effectuée *avant* tout calcul statistique pour éviter la fuite de données (*data leakage*). L'option `stratify=y` garantit de conserver les proportions de chaque classe.
2. **Encodage des étiquettes (One-Hot Encoding)** : Transformation des classes $(0, 1, 2)$ en vecteurs binaires.
3. **Normalisation (StandardScaler)** : Ajusté uniquement sur le jeu d'entraînement (`fit_transform`) puis appliqué sur le test (`transform`).

## 2. Architecture du Modèle (4 - 8 - 3)
* **Couche d'entrée** : 4 neurones (les 4 features).
* **Couche cachée** : 8 neurones avec une fonction d'activation **ReLU** (40 paramètres).
* **Couche de sortie** : 3 neurones avec une activation **Softmax** pour prédire les probabilités des 3 classes (27 paramètres).

## 3. Compilation et Entraînement
* **Optimizer** : `adam`.
* **Loss** : `categorical_crossentropy`.
* Évaluation finale et analyse de la matrice de confusion.

---
> *Code source du notebook et visualisations du graphe de neurones à venir.*