# TP4 : Réseaux de Neurones Convolutifs (CNN) - MNIST & Fashion-MNIST

## Objectif
Maîtriser les réseaux convolutifs pour la classification d'images en deux volets :

1. **MNIST** : Reconnaissance de chiffres manuscrits (0 à 9).

2. **Fashion-MNIST** : Classification d'articles de vestimentaires (10 catégories).

---

## 1. Prétraitement des images

* **Normalisation des pixels** : Division par 255.0 pour ramener les valeurs de $[0, 255]$ à $[0, 1]$.

* **Ajout du canal** : Passage d'une forme de $(N, 28, 28)$ à $(N, 28, 28, 1)$ pour expliciter le canal de nuances de gris.

## 2. Architecture type d'un CNN

* **Conv2D** : Extraction de motifs locaux (bords, textures).

* **MaxPooling2D** : Réduction spatiale de la dimension des cartes de caractéristiques (pooling).

* **Flatten** : Transformation de la matrice en vecteur plat.

* **Dense & Dropout** : Couche dense de classification et coupure aléatoire de connexions pour lutter contre l'overfitting.

## 3. Analyse des performances

* Évaluation de la précision sur le jeu de test et identification des classes les plus fréquemment confondues via la matrice de confusion.

---
> *Code source des notebooks et explications détaillées des couches à venir.*