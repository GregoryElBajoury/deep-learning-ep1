# 1. Introduction au Deep Learning et à l'IA

## Définitions et hiérarchie

*   **Intelligence artificielle (IA)** : Toutes les techniques qui permettent à une machine de simuler l'intelligence humaine : résolution de problèmes, décision, compréhension du langage.
*   **Machine Learning (ML)** : La machine apprend à partir de données au lieu de suivre des règles écrites à la main.
*   **Deep Learning (DL)** : Réseaux de neurones à plusieurs couches (« deep » veut dire profond). Excellent sur les images, les sons, les textes.

### Où on le croise
*   **Voitures autonomes** : détection d'obstacles, lecture des panneaux, planification de trajectoire.
*   **Assistants vocaux** : Siri, Alexa et Google Assistant comprennent la parole grâce au DL.
*   **ChatGPT et les LLM** : modèles de langage fondés sur des réseaux profonds.
*   **Reconnaissance faciale** : déverrouillage de téléphone, identification.
*   **Recommandation** : Netflix, YouTube, Spotify personnalisent leurs suggestions.
*   **Santé** : aide au diagnostic, analyse d'imagerie médicale.
*   **Finance** : détection de fraude, trading algorithmique.

---

## 🎥 Vidéo : Introduction au Deep Learning (Machine Learnia)

<iframe width="560" height="315" src="https://www.youtube.com/embed/XUFLq6dKQok" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

---

## Machine Learning classique

### Principe
Le ML cherche une fonction $f(x)$ qui approxime la relation entre les entrées $x$ (ex: la surface d'une maison) et les sorties $y$ (son prix), de sorte que $f(x) \approx y$ sur toutes les données.

### Exemple : la régression linéaire
Le modèle le plus simple cherche la droite qui passe au plus près des points.
$f(x) = a \cdot x + b$
$a$ est la pente et $b$ l'ordonnée à l'origine. Le modèle trouve les $a$ et $b$ qui rapprochent la droite de tous les points.

### Fonction de coût
Elle mesure l'erreur du modèle : plus elle est grande, plus le modèle est mauvais. Pour la régression, on utilise souvent l'erreur quadratique moyenne :
$L = (1/n) \cdot \Sigma (y_{réel} - y_{prédit})^2$

### Descente de gradient
C'est l'algorithme qui minimise la fonction de coût en ajustant les paramètres petit à petit :
1. Calculer l'erreur du modèle.
2. Calculer dans quelle direction modifier les paramètres pour la réduire (le gradient).
3. Faire un petit pas dans cette direction.
4. Répéter jusqu'à convergence.
`paramètre_nouveau = paramètre_ancien - learning_rate * gradient`

### Autres modèles classiques
*   **Arbres de décision** : des règles « si condition A alors X, sinon Y ». Faciles à interpréter, mais ils sur-apprennent.
*   **SVM** : cherchent l'hyperplan qui sépare au mieux les classes. Très efficaces en classification binaire.
*   **KNN** : classe un point selon la classe majoritaire de ses k plus proches voisins.
*   **Random Forest** : un ensemble d'arbres qui votent pour donner une prédiction plus robuste.

---

## Histoire des réseaux de neurones

*   **1943 - McCulloch et Pitts** : Premier modèle mathématique de neurone artificiel.
*   **1957 - Rosenblatt** : Invente le perceptron (premier algorithme capable d'apprendre automatiquement à partir de données sur des problèmes linéairement séparables).
*   **1969 - Minsky et Papert** : Démontrent qu'un perceptron simple ne résout pas XOR. (Premier hiver de l'IA).
*   **1986 - Rétropropagation** : Hinton popularise l'algorithme pour entraîner des réseaux multicouches (MLP). XOR est résolu.
*   **1989-1998 - LeCun et les CNN** : Réseaux convolutifs pour les images (ex: LeNet-5).
*   **Années 1990 - RNN** : Réseaux récurrents pour les séquences (texte, son).
*   **2012 - AlexNet** : Un réseau de 8 couches gagne ImageNet. (GPU, Big Data, ReLU, Dropout).
*   **2014-2015 - VGG, Inception, ResNet** : Réseaux toujours plus profonds.
*   **2017-2018 - Transformers, BERT, GPT** : Mécanisme d'attention pour le NLP.
*   **2020 et après - GPT-3, DALL-E, ChatGPT** : Modèles géants.