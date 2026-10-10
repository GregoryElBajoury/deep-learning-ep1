# TP4 : Réseaux de Neurones Convolutifs (CNN) - MNIST & Fashion-MNIST

## Objectif
Maîtriser les réseaux convolutifs pour la classification d'images en deux volets :
1. **MNIST** : Reconnaissance de chiffres manuscrits (0 à 9)[cite: 93, 120].
2. **Fashion-MNIST** : Classification d'articles de vestimentaires (10 catégories)[cite: 94, 124].

---

## 1. Prétraitement des images
* **Normalisation des pixels** : Division par 255.0 pour ramener les valeurs de $[0, 255]$ à $[0, 1]$[cite: 93, 120].
* **Ajout du canal** : Passage d'une forme de $(N, 28, 28)$ à $(N, 28, 28, 1)$ pour expliciter le canal de nuances de gris[cite: 93, 120].

## 2. Architecture type d'un CNN
* **Conv2D** : Extraction de motifs locaux (bords, textures)[cite: 76, 93].
* **MaxPooling2D** : Réduction spatiale de la dimension des cartes de caractéristiques (pooling)[cite: 93].
* **Flatten** : Transformation de la matrice en vecteur plat[cite: 93].
* **Dense & Dropout** : Couche dense de classification et coupure aléatoire de connexions pour lutter contre l'overfitting[cite: 77, 93].

## 3. Analyse des performances
* Évaluation de la précision sur le jeu de test et identification des classes les plus fréquemment confondues via la matrice de confusion[cite: 94, 124].

---
> *Code source des notebooks et explications détaillées des couches à venir.*