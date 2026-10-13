# 1. Introduction au Deep Learning et à l'IA

> Un support de cours interactif : entraîne un neurone, déplace une fonction d'activation et compare des optimizers directement depuis la page.

!!! abstract "Objectifs d'apprentissage"
    À la fin de ce chapitre, tu sauras :
- Situer l'IA, le Machine Learning et le Deep Learning les uns par rapport aux autres.
- Expliquer le fonctionnement d'un neurone artificiel et d'un réseau multicouche.
- Choisir une fonction d'activation, un optimizer et un learning rate adaptés.
- Entraîner un réseau avec TensorFlow/Keras en évitant toute fuite de données (*data leakage*).

---

## Un neurone qui apprend

### Les portes logiques
Une porte logique est une règle de décision élémentaire : elle reçoit deux signaux booléens ($0$ ou $1$) et produit une unique sortie. Le tableau ci-dessous résume les cas pour chaque configuration d'entrées ($x_1, x_2$) :

| Porte | La sortie vaut 1 quand… | (0, 0) | (0, 1) | (1, 0) | (1, 1) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **AND** (ET) | les deux entrées valent 1 | 0 | 0 | 0 | 1 |
| **OR** (OU) | au moins une entrée vaut 1 | 0 | 1 | 1 | 1 |
| **NOR** (NON-OU) | aucune entrée ne vaut 1 : l'inverse de OR | 1 | 0 | 0 | 0|
| **XOR** (OU exclusif) | une seule des deux entrées vaut 1| 0 | 1 | 1 | 0 |

!!! info "Le saviez-vous ?"
    Avec deux entrées, il existe $2 \times 2 \times 2 \times 2 = 16$ règles logiques possibles[cite: 11]. Sur ces 16 règles, **14 peuvent être résolues par un neurone unique** (car une simple droite suffit à séparer les classes)[cite: 11]. Seules **XOR** et **XNOR** nécessitent un réseau multicouches.

---

## 1. Introduction à l'IA, au Machine Learning et au Deep Learning

### Hiérarchie des concepts

- **Intelligence Artificielle (IA)** : L'ensemble des techniques permettant à une machine de simuler des traits de l'intelligence humaine (résolution de problèmes, prise de décision, raisonnement).
- **Machine Learning (ML)** : Un sous-domaine où le système apprend directement à partir de données historiques au lieu d'exécuter des instructions codées manuellement.
- **Deep Learning (DL)** : L'utilisation de réseaux de neurones artificiels profonds (« deep »)[cite: 11]. Ses performances sont particulièrement redoutables pour traiter des flux complexes (images, sons, texte brut).

## 🎥 Vidéo : Introduction au Deep Learning (Machine Learnia)

<div class="video-container" style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; border-radius: 8px; margin: 20px 0;">
  <iframe src="https://www.youtube.com/embed/XUFLq6dKQok" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
</div>

### Domaines d'application
- **Voitures autonomes** : perception de l'environnement, détection d'obstacles et planification de trajectoires.
- **Assistants vocaux** : reconnaissance et synthèse de la parole (Siri, Alexa, Google Assistant).
- **Traitement du langage** : modèles de type LLM et génération de texte (ChatGPT).
- **Vision et recommandation** : identification faciale, suggestions personnalisées (Netflix, YouTube, Spotify).
- **Santé et Finance** : aide au diagnostic par imagerie médicale et détection algorithmique des fraudes.

---

## 2. Machine Learning classique

### Principe
Le ML cherche une fonction mathématique $f(x)$ qui approxime au mieux la relation entre les entrées $x$ (ex: surface d'un bien immobilier) et les sorties $y$ (son prix), de sorte que $f(x) \approx y$.

### Exemple : la régression linéaire
Le modèle cherche à tracer la droite s'approchant le plus de l'ensemble des points :

$$f(x) = a \cdot x + b$$

Où $a$ est la pente et $b$ l'ordonnée à l'origine.

### Fonction de coût et Descente de gradient
Pour évaluer l'erreur, on utilise une fonction de coût (comme l'erreur quadratique moyenne - MSE):

$$L = \frac{1}{n} \sum (y_{\text{réel}} - y_{\text{prédit}})^2$$

La **descente de gradient** ajuste ensuite les paramètres de manière itérative pour minimiser cette erreur :

$$\text{paramètre}_{\text{nouveau}} = \text{paramètre}_{\text{ancien}} - \eta \times \text{gradient}$$

---

## 3. Histoire des réseaux de neurones

- **1943** : **McCulloch et Pitts** conçoivent le tout premier modèle mathématique de neurone artificiel.
- **1957** : **Frank Rosenblatt** invente le **perceptron**, premier algorithme d'apprentissage automatique.
- **1969** : **Minsky et Papert** démontrent l'incapacité du perceptron simple à résoudre le problème non-linéaire du XOR $\rightarrow$ Début du **premier hiver de l'IA** (1969-1980).
- **1986** : Popularisation de la **rétropropagation du gradient** par Hinton, permettant d'entraîner des réseaux multicouches et de surmonter le verrou du XOR.
- **1989-1998** : **Yann LeCun** introduit les réseaux de neurones convolutifs (**CNN**) avec le modèle *LeNet-5*.
- **2012** : **AlexNet** pulvérise les scores de la compétition ImageNet grâce au triptyque : puissance des **GPU**, volumes massifs de données et généralisation de la fonction **ReLU**.
- **2017-2018** : L'avènement des **Transformers** (mécanisme d'attention) avec BERT et GPT.
- **2020+** : Ère des modèles génératifs de grande taille (GPT-3, ChatGPT, multimodalité).

---

## 4. Le neurone artificiel et le perceptron

Un neurone artificiel réalise une combinaison linéaire de ses entrées suivie d'une transformation non-linéaire :

$$z = \sum (w_i x_i) + b \quad \text{et} \quad y = f(z)$$

### Les fonctions d'activation majeures
- **Sigmoïde** : Comprise dans $[0, 1]$. Idéale en sortie de classification binaire, mais sujette au phénomène de *vanishing gradient* (gradient saturé).
- **Tanh** : Comprise dans $[-1, 1]$, centrée en zéro.
- **ReLU ($\max(0, x)$)** : Le standard incontournable des couches cachées (CNN, MLP). Attention toutefois au syndrome du *Dying ReLU* (neurones bloqués à zéro).
- **Softmax** : Transforme des scores bruts en une distribution de probabilités dont la somme vaut $1$, idéale pour la classification multi-classes.

---

## 5. Réseaux de neurones multicouches (MLP)

Un réseau multicouche s'articule autour de trois types de structures :

- **Couche d'entrée** : Reçoit les données brutes sans effectuer de calcul.
- **Couches cachées** : Extraient des représentations hiérarchiques de plus en plus complexes (des bords bas niveau jusqu'aux objets complets).
- **Couche de sortie** : Formule la prédiction finale du modèle.

L'apprentissage s'appuie sur la **forward propagation** (calcul des prédictions) couplée à la **rétropropagation** (propagation de l'erreur via la règle de la chaîne pour actualiser les poids de chaque couche).

---

## 6. Les optimizers et le Learning Rate

- **SGD** : Descente de gradient stochastique par mini-batch (simple mais trajectoire bruitée).
- **Momentum** : Intègre de l'inertie pour stabiliser la descente et franchir les zones de stagnation.
- **RMSProp** : Ajuste dynamiquement le pas d'apprentissage de chaque paramètre individuellement.
- **Adam** : Associe les avantages de Momentum et de RMSProp. C'est le **standard actuel** de l'industrie par sa robustesse.

!!! warning "Le rôle critique du Learning Rate ($\eta$)"
    - **Trop petit** : L'entraînement piétine et la convergence est extrêmement lente.
    - **Trop grand** : Les pas d'optimisation sautent par-dessus la vallée, provoquant une divergence totale (valeurs `NaN`).
    - **Bien réglé** : La descente de la fonction de coût s'opère de manière fluide et rapide.

---

## 💡 Travaux Pratiques associés

Rends-toi dans la section **Travaux Pratiques** pour mettre les mains dans le cambouis sur tes mini-projets (du Perceptron sous NumPy aux CNN sur MNIST/Fashion-MNIST et génération de texte par RNN/LSTM).