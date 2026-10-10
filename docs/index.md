# Accueil - Introduction au Deep Learning

Bienvenue sur la documentation du projet **Introduction au Deep Learning**. 

Ce support théorique et pratique accompagne la réalisation des travaux dirigés visant à maîtriser les architectures de réseaux de neurones (MLP, CNN, RNN) avec TensorFlow et Keras.

---

##  Contexte et Enjeux du Projet

Après avoir découvert les fondements du Deep Learning (neurones, fonctions d'activation, fonction de perte, descente de gradient et optimiseurs), l'objectif est d'expérimenter sur plusieurs jeux de données de natures différentes : **tabulaires, images et texte**.

L'enjeu principal n'est pas seulement d'exécuter du code, mais de **comprendre pourquoi un modèle fonctionne ou échoue**. Les objectifs spécifiques sont de :

* Choisir une architecture réseau adaptée à la typologie de la donnée.
* Éviter la fuite de données (data leakage) entre l'entraînement et le test (ex: ajustement des scalers).
* Repérer et traiter le surapprentissage (overfitting).
* Analyser et interpréter les courbes d'apprentissage (loss et accuracy).

---

##  Les 6 Mini-Projets (Travaux Pratiques)

Le projet s'articule autour de 6 exercices progressifs :

1. **Le Perceptron simple** : Implémentation manuelle (NumPy) de l'apprentissage des portes logiques (AND, NOR) par descente de gradient.
2. **Classification multi-classes (Iris)** : Création d'un réseau multicouches (MLP) simple pour classifier des fleurs à partir de variables numériques.
3. **Classification binaire (Titanic)** : Prédiction de la survie de passagers à partir de données tabulaires hétérogènes nécessitant un prétraitement complet (valeurs manquantes, encodage, normalisation).
4. **Réseau Convolutif (CNN) sur MNIST** : Classification d'images (28x28) de chiffres manuscrits.
5. **Réseau Convolutif (CNN) sur Fashion-MNIST** : Classification d'images d'articles de mode pour comparer les performances par classe.
6. **Réseau Récurrent (RNN / LSTM)** : Modélisation séquentielle et génération de texte à partir d'un corpus de poèmes de Victor Hugo.

---

##  Sommaire de la Documentation Théorique

Pour t'accompagner dans la réalisation de ces notebooks, cette documentation regroupe les concepts théoriques clés illustrés par des ressources vidéo :

* [**1. Introduction au Deep Learning**](1-introduction-dl.md) : Hiérarchie de l'IA, Machine Learning vs Deep Learning, et historique.
* [**2. Le neurone et le perceptron**](2-neurone-perceptron.md) : Anatomie d'un neurone artificiel, fonctions d'activation (Sigmoïde, ReLU, Softmax) et limites.
* [**3. Réseaux multicouches (MLP)**](3-reseaux-multicouches.md) : Propagation, rétropropagation, le rôle des optimiseurs (SGD, Adam) et l'importance du Learning Rate.

---

##  Modalités et Livrables

* **Compétences visées** : Concevoir des modèles d'apprentissage profond, optimiser les hyperparamètres, effectuer l'analyse exploratoire.
* **Livrables attendus** : 
    1. Un **Notebook Jupyter/Colab** complet, commenté et fonctionnel couvrant les 6 exercices.
    2. Un **support de présentation** expliquant clairement le fonctionnement des modèles (notamment le détail des couches d'un CNN : Conv2D, MaxPooling, Flatten, Dense, Dropout).