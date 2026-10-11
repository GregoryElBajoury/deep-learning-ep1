# TP1 : Le Perceptron simple et les portes logiques

## Objectif
Implémenter un neurone artificiel (perceptron simple) en Python pur (`NumPy`) pour résoudre des problèmes de classification binaire, notamment l'apprentissage des portes logiques **AND** et **NOR** par descente de gradient.

---

## 1. Données d'entraînement (Portes AND & NOR)

Le perceptron prend en entrée une matrice $X$ de dimensions $(4, 2)$ représentant les quatre combinaisons possibles pour deux variables booléennes, et un vecteur cible $y$ associé.

```python
import numpy as np
import pandas as pd

# Entrées (X) : 4 cas possibles pour deux entrées (x1, x2)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Sortie attendue (Porte NOR : 1 uniquement si [0, 0], sinon 0)
y = np.array([1, 0, 0, 0])
```


## 2. Implémentation du neurone : Initialisation et Activation

Le neurone calcule la somme pondérée de ses entrées à laquelle il ajoute un biais, puis applique une fonction d'activation non-linéaire (la **sigmoïde**).

### Initialisation des paramètres
```python
np.random.seed(42)
w = np.random.randn(2)  # Poids initiaux
b = np.random.randn()   # Biais initial
```

### Fonction d'activation (Sigmoïde)

La sigmoïde comprime toute valeur réelle en une probabilité comprise entre $0$ et $1$.

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict(X, w, b):
    z = np.dot(X, w) + b
    return sigmoid(z)
```

## 3. Entraînement et Descente de Gradient

### Pourquoi l'Entropie Croisée Binaire (BCE) plutôt que la MSE ?
Pour des tâches logiques utilisant la sigmoïde, l'utilisation de l'Erreur Quadratique Moyenne (MSE) risque de provoquer une saturation du gradient. On privilégie donc l'**Entropie Croisée Binaire (BCE)**, qui offre un meilleur comportement d'apprentissage.

```python
def bce(y_true, y_pred, eps=1e-12):
    return -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))
```

### Boucle d'entraînement

```python
learning_rate = 0.5
epochs = 2000

for i in range(epochs):
    y_pred = predict(X, w, b)
    
    # Gradient simplifié de la BCE avec la sigmoïde
    dz = y_pred - y
    
    w -= learning_rate * np.dot(X.T, dz) / len(X)
    b -= learning_rate * np.mean(dz)
    
    if i % 400 == 0:
        print(f"Epoch {i}, Loss: {bce(y, y_pred):.4f}")

# Prédictions finales
final_preds = predict(X, w, b)
print("Prédictions finales (probabilités) :", final_preds.round(3))
print("Classes prédites (seuil 0.5)       :", (final_preds > 0.5).astype(int))
```

---

##  Ressources du TP

Retrouve l'intégralité du code et des expérimentations dans les notebooks source :

- 🔗 [Notebook - Porte AND (`01_perceptron_AND.ipynb`)](../../notebooks/01_perceptron_AND.ipynb)
- 🔗 [Notebook - Porte NOR (`01_perceptron_NOR.ipynb`)](../../notebooks/01_perceptron_NOR.ipynb)

!!! warning "Attention"
    Si ton neurone commence à refuser d'exécuter tes ordres ou à procrastiner à la place de la porte NOR, ne panique pas : c'est qu'il commence à développer une conscience artificielle... ou qu'il y a un bug dans ton `learning_rate`. Dans le doute, relance un café et un `git commit`. ☕🤖