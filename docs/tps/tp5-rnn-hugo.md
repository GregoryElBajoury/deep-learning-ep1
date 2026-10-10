# TP5 : Génération de texte séquentielle avec les RNN et LSTM

## Objectif
Modéliser des séquences textuelles à partir d'un corpus d'œuvres de Victor Hugo (`textes.txt`) pour prédire le mot suivant et générer de la poésie de manière autorégressive[cite: 95, 136].

---

## 1. Traitement textuel et Vectorisation
* **Tokenizer** : Conversion des mots en indices numériques basés sur un vocabulaire restreint (`vocab_size = 1000`)[cite: 95, 136].
* **Fenêtrage temporel (`window_size = 3`)** : Constitution des séquences d'apprentissage (les 3 mots précédents prédisent le 4e)[cite: 95, 136].
* **Pad_sequences** : Complétion par des zéros (*bourrage*) pour uniformiser la taille des séquences d'entrée[cite: 96, 136].

## 2. Architecture du Modèle Séquentiel
* **Embedding** : Plongeon lexical pour projeter les indices de mots en vecteurs denses continus (`embedding_dim = 16`)[cite: 96, 137].
* **SimpleRNN vs LSTM** : Comparaison des performances et de la mémoire de contexte à long terme en remplaçant la couche `SimpleRNN(32)` par un `LSTM(64)`[cite: 96, 138].
* **Couche de sortie** : Dense avec activation `softmax` sur tout le vocabulaire[cite: 96, 137].

## 3. Génération de texte et Température
* Fonction de génération autorégressive utilisant un tirage pondéré modulé par un paramètre de **température** (pour ajuster la créativité ou la répétitivité du texte généré)[cite: 97, 138].

---
> *Code source du notebook et exemples de textes générés à venir.*