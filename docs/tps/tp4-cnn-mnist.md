# TP4 : Réseaux de Neurones Convolutifs (CNN) - MNIST & Fashion-MNIST

## Objectif

Ce TP a pour but de te faire maîtriser les Réseaux de Neurones Convolutifs (CNN) pour la classification d'images à travers deux jeux de données classiques :

1. **MNIST** : Reconnaissance de chiffres manuscrits ($0$ à $9$).
2. **Fashion-MNIST** : Classification d'articles vestimentaires ($10$ catégories).

---

## 1. Étapes clés du pipeline de Deep Learning

* **Prétraitement des images** :
  * Normalisation des pixels (division par $255.0$ pour ramener les valeurs de $[0, 255]$ à $[0, 1]$).
  * Redimensionnement/Ajout de dimension du canal (passage de $(N, 28, 28)$ à $(N, 28, 28, 1)$ pour la couche convolutionnelle).

* **Architecture d'un CNN** :
  * `Conv2D` : Extraction de caractéristiques spatiales (bords, textures).
  * `MaxPooling2D` : Réduction de la dimension spatiale (downsampling).
  * `Flatten` : Aplatissement des cartes de caractéristiques en un vecteur 1D.
  * `Dense` : Couches de classification entièrement connectées.
  * `Dropout` : Régularisation pour éviter le surapprentissage (overfitting).

* **Évaluation** :
  * Calcul de la précision (`accuracy`) et de la perte (`loss`).
  * Analyse par matrice de confusion et rapport de classification (`classification_report`).

---

## 2. Volet 1 : Application sur le dataset MNIST

Le dataset MNIST contient des images de chiffres de $0$ à $9$ en nuances de gris ($28 \times 28$ pixels). 

### Code d'implémentation pour MNIST :

```
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Input
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement des données MNIST
(X_train_mnist, y_train_mnist), (X_test_mnist, y_test_mnist) = datasets.mnist.load_data()

# Prétraitement
X_train_mnist = X_train_mnist.astype("float32") / 255.0
X_test_mnist = X_test_mnist.astype("float32") / 255.0

# Ajout du canal
X_train_mnist = np.expand_dims(X_train_mnist, -1)
X_test_mnist = np.expand_dims(X_test_mnist, -1)

# Construction du modèle CNN pour MNIST
model_mnist = models.Sequential([
    Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model_mnist.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entraînement
history_mnist = model_mnist.fit(X_train_mnist, y_train_mnist, epochs=5, batch_size=64, validation_split=0.1)

# Évaluation
test_loss, test_acc = model_mnist.evaluate(X_test_mnist, y_test_mnist, verbose=2)
print(f"Précision MNIST sur le test : {test_acc * 100:.2f}%")
```

---

## 3. Volet 2 : Application sur le dataset Fashion-MNIST

Fashion-MNIST remplace MNIST pour complexifier la tâche avec 10 catégories de vêtements (T-shirt, pantalon, pull, etc.). Les pixels partagent la même dimension ($28 \times 28 \times 1$), mais les formes texturées y sont plus complexes.

### Observations et Comparaison
- Les performances sur MNIST atteignent rapidement des sommets (souvent $> 98\%$ ou $99\%$ de précision dès quelques époques).
- Sur Fashion-MNIST, la précision se stabilise généralement autour de $90\% - 92\%$ en raison de la complexité visuelle accrue et des ressemblances entre certaines classes (ex: chemises et t-shirts).

---

## Ressources du TP

Retrouve l'intégralité du code et des expérimentations dans le notebook source sur GitHub :

## Ressources du TP

Retrouve l'intégralité du code et des expérimentations dans les notebooks sources sur GitHub :

- 🔗 [Notebook - Réseau de neurones CNN - Dataset MNIST (`03_apprentissage_CNN_dataset_MNIST.ipynb`)](https://github.com/GregoryElBajoury/deep-learning-ep1/blob/main/notebooks/03_apprentissage_CNN_dataset_MNIST.ipynb)
- 🔗 [Notebook - Réseau de neurones CNN - Dataset Fashion-MNIST (`04_apprentissage_CNN_dataset_Fashion_MNIST.ipynb`)](https://github.com/GregoryElBajoury/deep-learning-ep1/blob/main/notebooks/04_apprentissage_CNN_dataset_Fashion_MNIST.ipynb)
---