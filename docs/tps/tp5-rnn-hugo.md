# TP5 (Pratique) : Génération de texte séquentielle avec les RNN et LSTM

## Objectif
Modéliser des séquences textuelles à partir d'un corpus d'œuvres de Victor Hugo (`textes.txt`) pour prédire le mot suivant et générer de la poésie de manière autorégressive.

---

## 1. Traitement textuel et Vectorisation
- **Tokenizer** : Conversion des mots en indices numériques basés sur un vocabulaire restreint (`vocab_size = 1000`).
- **Fenêtrage temporel (`window_size = 3`)** : Constitution des séquences d'apprentissage (les 3 mots précédents prédisent le 4e).
- **Pad_sequences** : Complétion par des zéros (*bourrage*) pour uniformiser la taille des séquences d'entrée.

## 2. Architecture du Modèle Séquentiel
- **Embedding** : Plongeon lexical pour projeter les indices de mots en vecteurs denses continus (`embedding_dim = 16`).
- **SimpleRNN vs LSTM** : Comparaison des performances et de la mémoire de contexte à long terme en remplaçant la couche `SimpleRNN(32)` par un `LSTM(64)`.
- **Couche de sortie** : Dense avec activation `softmax` sur tout le vocabulaire.

## 3. Génération de texte et Température
- Fonction de génération autorégressive utilisant un tirage pondéré modulé par un paramètre de **température** (pour ajuster la créativité ou la répétitivité du texte généré).

---

## Synthèse et Questions Clés (FAQ du TP)

### 1. Qu'est-ce que fait le `Tokenizer` et pourquoi est-il nécessaire dans le traitement de texte avant un RNN ?
* **Rôle** : Le `Tokenizer` analyse le corpus textuel, extrait l'ensemble des mots uniques pour constituer un dictionnaire (le vocabulaire), puis convertit chaque mot en une suite d'indices numériques (entiers).
* **Nécessité** : Les réseaux de neurones (et les mathématiques sous-jacentes) ne comprennent pas les chaînes de caractères brutes ; ils ont besoin de tableaux de nombres pour effectuer leurs calculs matriciels.

### 2. Pourquoi utilise-t-on `pad_sequences` et que se passerait-il si on ne l'utilisait pas ?
* **Rôle** : `pad_sequences` permet de standardiser la longueur de toutes les séquences d'entrée en ajoutant du bourrage (*padding* à zéros) pour qu'elles aient toutes exactement la même taille (`window_size`).
* **Conséquence sans padding** : TensorFlow exige des tenseurs de dimensions rectangulaires (uniformes). Si les séquences avaient des tailles différentes, le modèle ne pourrait pas empiler les données en *batches* (lots) homogènes pour l'entraînement, ce qui déclencherait une erreur de dimension.

### 3. Quelle est la différence entre `vocab_size` et `embedding_dim` dans ce modèle ?
* **`vocab_size`** : Représente le nombre total de mots uniques pris en compte dans notre dictionnaire (ex: 1000 mots les plus fréquents).
* **`embedding_dim`** : Représente la taille de l'espace vectoriel dans lequel chaque mot est projeté (ex: 16 dimensions). C'est la taille de la "carte d'identité" sémantique apprise pour chaque mot.

### 4. Explique en quoi consiste une couche `Embedding` et pourquoi on l'utilise avant le RNN.
* **Fonctionnement** : La couche `Embedding` agit comme une table de correspondance dynamique qui transforme un indice entier (le numéro du mot) en un vecteur de nombres continus de taille `embedding_dim`.
* **Utilisation** : Elle permet de capturer des relations sémantiques entre les mots (des mots proches se retrouveront proches dans cet espace vectoriel), ce qui est beaucoup plus riche pour le réseau qu'un simple encodage *one-hot* brut et creux.

### 5. Quelle est la différence entre `SimpleRNN` et `LSTM` ? Dans quel cas préférerait-on un LSTM ?
* **`SimpleRNN`** : Met à jour son état caché de manière directe à chaque pas de temps, ce qui le rend incapable de retenir des informations sur le long terme à cause du phénomène de disparition du gradient (*vanishing gradient*).
* **`LSTM` (*Long Short-Term Memory*)** : Intègre un mécanisme de portes (*gates*: oubli, entrée, sortie) et une cellule mémoire dédiée qui régulent le flux d'informations.
* **Préférence** : On préfère toujours le **LSTM** dès que l'on traite des séquences un tant soit peu longues pour préserver le contexte global.

### 6. Que représente `window_size=3` et comment cela influence-t-il l'apprentissage du modèle ?
* **Définition** : Cela signifie que le modèle regarde les **3 mots précédents** pour prédire le 4e.
* **Influence** : Un `window_size` petit limite le modèle à un contexte purement local (il retient l'accord immédiat mais oublie le début de la phrase). L'augmenter donne plus de contexte historique au réseau, mais alourdit l'apprentissage.

### 7. Pourquoi utilise-t-on une fonction de perte (`sparse_categorical_crossentropy`) dans ce cas précis ?
* **Cas d'usage** : Nos étiquettes cibles (`y`) sont des indices entiers représentant le mot suivant, et la couche de sortie (`Dense` avec une activation `softmax`) produit une distribution de probabilités sur tout le vocabulaire.
* **Utilité** : `sparse_categorical_crossentropy` est spécialement conçue pour ce cas de classification multiclasse où les classes cibles sont sous forme d'entiers.

### 8. Que fait le callback `EarlyStopping`, et pourquoi est-il utile ici ?
* **Rôle** : Il surveille une métrique (comme la perte) pendant l'entraînement et interrompt automatiquement l'algorithme si la métrique cesse de s'améliorer après un certain nombre d'époques (`patience`), tout en pouvant restaurer les meilleurs poids (`restore_best_weights=True`).
* **Utilité** : Il évite le surapprentissage (*overfitting*) et fait gagner un temps précieux en évitant d'entraîner le modèle inutilement.

### 9. En observant la courbe de la loss pendant l'entraînement, comment peut-on savoir si le modèle converge bien ?
* Une bonne convergence se traduit par une **diminution régulière et asymptotique** de la perte vers une valeur basse (proche de 0), tout en se stabilisant. Si la *loss* descend puis commence à remonter, cela indique un surapprentissage.

### 10. Analyse le texte généré : est-il grammaticalement correct et cohérent ? Pourquoi le modèle n'arrive-t-il pas encore à écrire comme Victor Hugo ?
* **Analyse** : Le texte montre de belles fulgurances locales et un vocabulaire soutenu, mais finit par perdre le fil syntaxique sur la durée.
* **Limites** : Le modèle reste un simple processeur statistique de court/moyen terme. Il n'a aucune conscience de la structure narrative globale ni de la syntaxe littéraire complexe de Victor Hugo ; il calcule uniquement des probabilités de transition de mots locaux.

---

## Ressources du TP

Retrouve l'intégralité du code et des expérimentations dans les notebooks sources sur GitHub :

- 🔗 [Notebook - Réseau de neurones RNN & Texte Victor Hugo (`06_apprentissage_RNN_Corpus_Victor_Hugo.ipynb`)](https://github.com/GregoryElBajoury/deep-learning-ep1/blob/main/notebooks/6_apprentissage_RNN_Corpus_Victor_Hugo.ipynb)
- 🔗 [Notebook - Questions / Réponses de Synthèse (`07_questions_generation_de_texte_Victor_Hugo.ipynb`)](https://github.com/GregoryElBajoury/deep-learning-ep1/blob/main/notebooks/7_questions_generation_de_texte_Victor_Hugo.ipynb)