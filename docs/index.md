# Accueil - Introduction au Deep Learning

Bienvenue sur la documentation officielle du projet **Introduction au Deep Learning**. 

Ce support technique et pratique accompagne la réalisation des travaux dirigés visant à maîtriser les architectures de réseaux de neurones profonds (MLP, CNN, RNN) avec TensorFlow et Keras.

---

##  Contexte et Enjeux du Projet

Après avoir découvert les fondements du Deep Learning (neurones, fonctions d'activation, fonctions de perte, descente de gradient et optimiseurs), l'objectif est d'expérimenter sur plusieurs jeux de données de natures différentes : **tabulaires, images et texte**.

!!! abstract "Objectif Pédagogique"
    L'enjeu principal n'est pas seulement d'exécuter du code, mais de **comprendre pourquoi un modèle fonctionne ou échoue**. Les axes majeurs de ce projet sont :
    * Choisir une architecture de réseau adaptée à la typologie de la donnée.
    * Éviter la fuite de données (*data leakage*) entre l'entraînement et le test (ex: ajustement rigoureux des scalers).
    * Repérer et traiter le surapprentissage (*overfitting*).
    * Analyser et interpréter finement les courbes d'apprentissage (`loss` et `accuracy`).

---

##  Les 6 Mini-Projets & Travaux Pratiques

Le parcours s'articule autour de 5 rubriques de TPs intégrées à cette documentation :

1. **[TP1 : Le Perceptron simple](tps/tp1-perceptron.md)** : Implémentation manuelle (NumPy) de l'apprentissage des portes logiques (`AND`, `NOR`) par descente de gradient.
2. **[TP2 : Classification Iris (MLP)](tps/tp2-iris-mlp.md)** : Création d'un réseau multicouches simple pour classifier des fleurs à partir de mesures morphologiques.
3. **[TP3 : Titanic (Tabulaire)](tps/tp3-titanic.md)** : Prédiction de survie de passagers à partir de données hétérogènes (prétraitement, imputations, normalisation).
4. **[TP4 : CNN (MNIST & Fashion)](tps/tp4-cnn-mnist.md)** : Classification d'images par réseaux convolutifs (chiffres manuscrits et articles de mode).
5. **[TP5 : RNN & Texte (Victor Hugo)](tps/tp5-rnn-hugo.md)** : Modélisation séquentielle et génération de poésie de manière autorégressive.

---

##  Sommaire de la Documentation Théorique

Pour réviser les notions clés avant de coder, consulte les fiches théoriques :

- [**1. Introduction au DL & IA**](1-introduction-dl.md) : Hiérarchie de l'IA, Machine Learning vs Deep Learning, et repères historiques.
- [**2. Le Neurone & Perceptron**](2-neurone-perceptron.md) : Anatomie d'un neurone artificiel, fonctions d'activation (`Sigmoïde`, `ReLU`, `Softmax`) et limites.
- [**3. Réseaux Multicouches (MLP)**](3-reseaux-multicouches.md) : Propagation, rétropropagation, rôle des optimiseurs (`SGD`, `Adam`) et gestion du *learning rate*.

---

##  Modalités et Livrables

!!! info "Attentes du brief"
    * **Compétences visées** : Concevoir des modèles d'apprentissage profond, optimiser les hyperparamètres, effectuer l'analyse exploratoire et interpréter les résultats.
    * **Livrables attendus** : 
        1. Un **Notebook Jupyter/Colab** complet, commenté et fonctionnel couvrant les exercices.
        2. Un **support de présentation** expliquant clairement le fonctionnement des modèles (notamment le détail des couches d'un CNN : `Conv2D`, `MaxPooling`, `Flatten`, `Dense`, `Dropout`).