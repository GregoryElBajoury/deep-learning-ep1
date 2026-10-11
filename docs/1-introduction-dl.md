# 1. Introduction au Deep Learning et à l'IA

> Un support de cours interactif : entraîne un neurone, déplace une fonction d'activation et compare des optimizers directement depuis la page.

!!! abstract "Objectifs d'apprentissage"
    À la fin de ce chapitre, tu sauras :
    * Situer l'IA, le Machine Learning et le Deep Learning les uns par rapport aux autres.
    * Expliquer le fonctionnement d'un neurone artificiel et d'un réseau multicouche.
    * Choisir une fonction d'activation, un optimizer et un learning rate adaptés.
    * Entraîner un réseau avec TensorFlow/Keras en évitant toute fuite de données (*data leakage*).

---

##  Un neurone qui apprend

### Les portes logiques
Une porte logique est une petite règle de décision : elle reçoit deux réponses « oui » ou « non » (1 pour oui, 0 pour non) et rend une seule réponse. Le tableau indique la sortie pour chaque combinaison d'entrées ($x_1, x_2$).

| Porte | La sortie vaut 1 quand… | (0, 0) | (0, 1) | (1, 0) | (1, 1) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **AND** (ET) | les deux entrées valent 1 | 0 | 0 | 0 | 1 |
| **OR** (OU) | au moins une entrée vaut 1 | 0 | 1 | 1 | 1 |
| **NOR** (NON-OU) | aucune entrée ne vaut 1 (l'inverse de OR) | 1 | 0 | 0 | 0 |
| **XOR** (OU exclusif) | une seule des deux entrées vaut 1 | 0 | 1 | 1 | 0 |

!!! info "Le saviez-vous ?"
    Avec deux entrées, il y a 16 règles possibles en tout. Sur ces 16, **14 se séparent par une droite** (un neurone seul les apprend, comme NAND ou XNOR). Seules XOR et XNOR demandent plusieurs neurones.

---

##  1. Introduction à l'IA, au Machine Learning et au Deep Learning

### Hiérarchie des concepts

* **Intelligence Artificielle (IA)** : Toutes les techniques qui permettent à une machine de simuler l'intelligence humaine (résolution de problèmes, décision, compréhension du langage).
    * **Machine Learning (ML)** : La machine apprend à partir de données au lieu de suivre des règles écrites à la main.
        * **Deep Learning (DL)** : Réseaux de neurones à plusieurs couches (*deep* = profond). Excellents sur les images, les sons et les textes.

## 🎥 Vidéo : Introduction au Deep Learning (Machine Learnia)

<div class="video-container" style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; border-radius: 8px; margin: 20px 0;">
  <iframe src="https://www.youtube.com/embed/XUFLq6dKQok" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
</div>



### Domaines d'application
* **Voitures autonomes** : détection d'obstacles, lecture de panneaux, planification de trajectoire.
* **Assistants vocaux** : Siri, Alexa et Google Assistant.
* **LLM & Génération** : ChatGPT, modèles de langage profonds.
* **Reconnaissance faciale** & **systèmes de recommandation** (Netflix, Spotify).
* **Santé** : aide au diagnostic et imagerie médicale.

---

##  2. Machine Learning classique

### Principe
Le ML cherche une fonction $f(x)$ qui approxime la relation entre les entrées $x$ (ex: surface d'une maison) et les sorties $y$ (son prix), de sorte que $f(x) \approx y$.

### Exemple : la régression linéaire
Le modèle cherche la droite qui passe au plus près des points :
$$f(x) = a \cdot x + b$$
Où $a$ est la pente et $b$ l'ordonnée à l'origine.

### Fonction de coût et Descente de gradient
La fonction de coût mesure l'erreur du modèle (ex: l'Erreur Quadratique Moyenne - MSE). La descente de gradient minimise cette erreur pas à pas :
$$\text{paramètre}_{\text{nouveau}} = \text{paramètre}_{\text{ancien}} - \eta \times \text{gradient}$$
*(où $\eta$ est le learning rate).*

---

##  3. Histoire des réseaux de neurones

* **1943** : McCulloch & Pitts (premier modèle mathématique).
* **1957** : Rosenblatt invente le **Perceptron**.
* **1969** : Minsky & Papert montrent les limites du perceptron (échec sur XOR) $\rightarrow$ Premier hiver de l'IA.
* **1986** : Popularisation de la **Rétropropagation** (Hinton) $\rightarrow$ Résolution du XOR.
* **1989-1998** : LeCun et les **CNN** (LeNet-5 pour les images).
* **2012** : **AlexNet** révolutionne ImageNet (GPU, Big Data, ReLU).
* **2017-2018** : Les **Transformers** (BERT, GPT) et le mécanisme d'attention.
* **2020+** : Modèles géants (GPT-3, ChatGPT, multimodalité).

---

##  4. Le neurone artificiel et le perceptron

Un neurone combine des entrées pondérées, ajoute un biais, puis applique une fonction d'activation non-linéaire :
$$z = \sum (w_i x_i) + b \quad \text{et} \quad y = f(z)$$

### Les fonctions d'activation majeures
* **Sigmoïde** : Sortie entre $[0, 1]$. Idéale en classification binaire, mais souffre du *vanishing gradient*.
* **Tanh** : Sortie entre $[-1, 1]$, centrée en zéro.
* **ReLU** ($\max(0, x)$) : Le standard incontournable des couches cachées (CNN, MLP). Attention au syndrome du *Dying ReLU*.
* **Softmax** : Utilisé en sortie pour transformer des scores bruts en probabilités multi-classes.

---

##  5. Réseaux de neurones multicouches (MLP)

Un MLP est composé d'une couche d'entrée, de **couches cachées** (pleinement connectées) et d'une couche de sortie. L'apprentissage s'effectue en deux phases principales :
1. **Forward propagation** : Calcul des prédictions de gauche à droite.
2. **Backpropagation (Rétropropagation)** : Remontée de l'erreur de droite à gauche via la règle de la chaîne pour ajuster chaque poids et chaque biais.

---

##  6. Les optimizers et le Learning Rate

* **SGD** : Simple mais lent et bruité.
* **Momentum** : Ajoute de l'inertie pour passer les petits creux.
* **RMSProp** : Adapte le pas de chaque paramètre individuellement.
* **Adam** : Combine Momentum et RMSProp. C'est le **standard actuel** de l'industrie.

!!! warning "Le rôle du Learning Rate ($\eta$)"
    * **Trop petit** : L'entraînement piétine et stagne.
    * **Trop grand** : La fonction de coût explose (*divergence* en valeurs `NaN`).
    * **Bien réglé** : La descente est fluide et rapide.

---

## 💡 Travaux Pratiques associés

Rends-toi dans la section **Travaux Pratiques** pour mettre les mains dans le cambouis sur 5 mini-projets concrets (du Perceptron sous NumPy aux CNN sur MNIST/Fashion-MNIST et génération de texte par RNN/LSTM).