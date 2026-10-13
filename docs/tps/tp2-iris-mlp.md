# TP2 (Pratique) : Implémentation et Entraînement d'un MLP avec Keras (Dataset Iris)

## Objectif
Mettre en œuvre un réseau de neurones multicouches (MLP) sous Keras/TensorFlow pour classifier les fleurs Iris en 3 catégories, en suivant un pipeline rigoureux (installation des dépendances, préparation, construction, compilation et entraînement).

---

## 1. Environnement et Dépendances
- **Bibliothèques clés** : `numpy`, `pandas`, `seaborn`, `matplotlib`, `networkx`, et `tensorflow` (avec Keras).
- **Reproductibilité** : Fixation de la graine aléatoire via `tf.keras.utils.set_random_seed(42)` pour garantir des résultats identiques à chaque exécution.

## 2. Chargement et Analyse Préliminaire des Données
1. **Chargement** : Utilisation du dataset `load_iris()` de scikit-learn.
   - **X (Features)** : Matrice de dimensions $(150, 4)$ correspondant aux 4 mesures morphologiques (longueur/largeur des sépales et pétales).
   - **y (Target)** : Vecteur de dimensions $(150,)$ représentant les 3 classes cibles ($0$ : Setosa, $1$ : Versicolor, $2$ : Virginica).
2. **Séparation Train/Test** : 
   - Utilisation de `train_test_split` avec un ratio de 20 % pour le test (`test_size=0.2`).
   - Application de l'option `stratify=y` pour conserver rigoureusement les proportions des 3 classes dans les ensembles d'entraînement et de test[cite: 12].
   - **Dimensions obtenues** : Train de $(120, 4)$ et Test de $(30, 4)$.

## 3. Prétraitement et Ingénierie des Données
1. **One-Hot Encoding (Étiquettes)** : 
   - Transformation des classes cibles entières en vecteurs binaires à l'aide de `OneHotEncoder` (ex: la classe `0` devient `[1., 0., 0.]`).
2. **Standardisation (Features)** : 
   - Utilisation de `StandardScaler` pour centrer et réduire les variables.
   - **Règle d'or respectée** : Le scaler est ajusté uniquement sur le jeu d'entraînement (`fit_transform`) puis appliqué tel quel sur le jeu de test (`transform`) pour éviter toute fuite de données (*data leakage*).

## 4. Architecture du Modèle MLP (4 - 8 - 3)
Création d'un modèle séquentiel (`Sequential`) comportant 67 paramètres entraînables au total :

- **Couche d'entrée** : Définie via `Input(shape=(4,))`.
- **Couche cachée (Dense)** : 8 neurones avec la fonction d'activation **ReLU** (générant 40 paramètres).
- **Couche de sortie (Dense)** : 3 neurones avec une activation **Softmax** pour convertir les scores en probabilités (27 paramètres).
- *Visualisation graphique* : Un graphe de connexions complet (généré via `networkx`) illustre la structure en couches (4 $\rightarrow$ 8 $\rightarrow$ 3).

## 5. Compilation et Entraînement
- **Compilateur** :
  - **Optimizer** : `adam`
  - **Loss** : `categorical_crossentropy` (adaptée au One-Hot Encoding multi-classes)
  - **Métrique** : `accuracy`
- **Entraînement (Fit)** : 
  - Exécuté sur 100 époques.
  - Suivi de la convergence de la précision (accuracy) et de la diminution de la fonction de perte (loss) à la fois sur l'entraînement et sur la validation.

---
> *Modèle pleinement opérationnel avec une convergence observée vers une précision optimale sur le jeu d'évaluation.*

##  Ressources du TP

Retrouve l'intégralité du code et des expérimentations dans le notebook source sur GitHub :

- 🔗 [Notebook - Réseau de neurones IRIS (`02_reseau_de_neurones_IRIS.ipynb`)](https://github.com/GregoryElBajoury/deep-learning-ep1/blob/main/notebooks/02_reseau_de_neurones_IRIS.ipynb)

---