# TP1 : Le Perceptron simple et les portes logiques (AND & NOR)

## Objectif

Implémenter un neurone artificiel (perceptron simple) en Python pur (`NumPy`) pour résoudre des problèmes de classification binaire, notamment l'apprentissage des portes logiques **AND** et **NOR** par descente de gradient.

## 1. Données d'entraînement

Le perceptron prend en entrée une matrice $X$ de dimensions $(4, 2)$ représentant les quatre combinaisons possibles pour deux variables booléennes, et un vecteur cible $y$ associé.

```
import numpy as np
import pandas as pd

# Entrées (X) : 4 cas possibles pour deux entrées (x1, x2)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Sorties cibles selon la porte logique étudiée :
# - Porte AND : 1 uniquement si [1, 1], sinon 0
# y_and = np.array([0, 0, 0, 1])

# - Porte NOR : 1 uniquement si [0, 0], sinon 0
y_nor = np.array([1, 0, 0, 0])

```


## 2. Initialisation et Fonctions Fondamentales

### Initialisation des paramètres

```
np.random.seed(42)
w = np.random.randn(2)  # Poids initiaux (un par entrée)
b = np.random.randn()   # Biais initial (seuil)
```

### Fonction d'activation (Sigmoïde)
La sigmoïde comprime toute valeur réelle en une probabilité comprise entre $0$ et $1$.

```
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict(X, w, b):
    z = np.dot(X, w) + b
    return sigmoid(z)
```

## 3. Choix de la Fonction de Perte (Loss)

Selon l'approche choisie, on utilise l'une ou l'autre des fonctions de perte :

### Option A : L'Erreur Quadratique Moyenne (MSE)
Idéale pour débuter et observer la descente de gradient pas à pas.

```
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
```

### Option B : L'Entropie Croisée Binaire (BCE)
Préférable pour des classifications avec sigmoïde afin d'éviter la saturation du gradient. Un paramètre `eps` est introduit pour éviter un `log(0)`.

```
def bce_loss(y_true, y_pred, eps=1e-12):
    return -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))
```


## 4. Entraînement et Descente de Gradient

### Exemple d'entraînement (Porte NOR avec BCE)

```
learning_rate = 0.5
epochs = 2000

for i in range(epochs):
    y_pred = predict(X, w, b)
    
    # Gradient simplifié de la BCE couplée à la sigmoïde
    dz = y_pred - y_nor
    
    w -= learning_rate * np.dot(X.T, dz) / len(X)
    b -= learning_rate * np.mean(dz)
    
    if i % 400 == 0:
        print(f"Epoch {i}, Loss: {bce_loss(y_nor, y_pred):.4f}")

# Prédictions finales
final_preds = predict(X, w, b)
print("Prédictions finales (probabilités) :", final_preds.round(3))
print("Classes prédites (seuil 0.5)       :", (final_preds > 0.5).astype(int))
```

## Ressources du TP

Retrouve l'intégralité du code et des expérimentations dans les notebooks source sur GitHub :

- 🔗 [Notebook - Porte AND (`01_perceptron_AND.ipynb`)](https://github.com/GregoryElBajoury/deep-learning-ep1/blob/main/notebooks/01_perceptron_AND.ipynb)
- 🔗 [Notebook - Porte NOR (`01_perceptron_NOR.ipynb`)](https://github.com/GregoryElBajoury/deep-learning-ep1/blob/main/notebooks/01_perceptron_NOR.ipynb)

!!! warning "Attention"
    Si ton neurone commence à refuser d'exécuter tes ordres ou à procrastiner à la place de la porte logique, ne panique pas : c'est qu'il commence à développer une conscience artificielle... ou qu'il y a un bug dans ton `learning_rate`. Dans le doute, relance un café et un `git commit`. ☕🤖