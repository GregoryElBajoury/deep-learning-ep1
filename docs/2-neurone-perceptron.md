# 2. Le neurone artificiel et le perceptron

## Anatomie d'un neurone

Un neurone artificiel est inspiré du neurone biologique et constitue la brique de base de tout réseau de neurones. Il est composé de :

- **Entrées ($x_1, x_2, \dots, x_n$)** : les données brutes ou les signaux provenant de la couche précédente.
- **Poids ($w_1, w_2, \dots, w_n$)** : les coefficients synaptiques qui pondèrent l'importance de chaque entrée. Un poids fort amplifie le signal, un poids faible le minore.
- **Biais ($b$)** : un paramètre additionnel qui permet de décaler la fonction d'activation, agissant comme un seuil de déclenchement personnalisé.

Le calcul s'effectue en deux étapes successives :

1. **La somme pondérée ($z$)** : 

   $$z = w_1x_1 + w_2x_2 + \dots + w_nx_n + b = \sum_{i=1}^{n}(w_i x_i) + b$$

2. **La fonction d'activation ($f$)** : elle introduit de la non-linéarité pour produire la sortie finale :

   $$y = f(z)$$

## 🎥 Vidéo : Le Perceptron (Machine Learnia)

<iframe width="560" height="315" src="https://www.youtube.com/embed/VlMm4VZ6lk4" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

---

## Un exemple concret à la main : La porte NOR

Pour bien comprendre comment agissent les poids et le biais, prenons un exemple simple : une alarme qui reste calme (sortie $1$) seulement si **aucun** des deux capteurs ne détecte de présence ($x_1$ et $x_2$ valent $1$ lorsqu'ils sont activés). C'est la table de vérité de la porte logique **NOR** (NON-OU).

Considérons un neurone dont les paramètres ont déjà été réglés ainsi :

* $w_1 = -2$
* $w_2 = -2$
* $b = +1$

Le neurone calcule $z = -2x_1 - 2x_2 + 1$ et renvoie $1$ si $z > 0$, sinon $0$. Suivons le calcul pour chaque combinaison :

| $x_1$ | $x_2$ | Calcul détaillé ($z = -2x_1 - 2x_2 + 1$) | Sortie du neurone ($z > 0$) | Sortie attendue (NOR) |
| :---: | :---: | :--- | :---: | :---: |
| **0** | **0** | $z = (-2\times0) + (-2\times0) + 1 = +1$ (positif) | **1** | **1** |
| **0** | **1** | $z = (-2\times0) + (-2\times1) + 1 = -1$ (négatif) | **0** | **0** |
| **1** | **0** | $z = (-2\times1) + (-2\times0) + 1 = -1$ (négatif) | **0** | **0** |
| **1** | **1** | $z = (-2\times1) + (-2\times1) + 1 = -3$ (négatif) | **0** | **0** |

**Ce qu'il faut retenir de cette logique :**

- Les **poids négatifs ($-2$)** font que chaque entrée active ($1$) tire le score global vers le bas.
- Le **biais positif ($+1$)** offre un point de départ favorable : en l'absence de signal extérieur, le neurone choisit par défaut de s'activer.

---

## Les fonctions d'activation

Elles sont indispensables car elles permettent au réseau de modéliser des relations non-linéaires complexes.

| Fonction | Sortie | Problèmes principaux | Utilisation typique |
| :--- | :--- | :--- | :--- |
| **Sigmoïde**<br>$\frac{1}{1+e^{-x}}$ | $[0, 1]$ | *Vanishing gradient* (gradient saturé aux extrêmes), non centrée en 0 | Sortie d'une classification binaire (probabilité) |
| **Tanh**<br>$\frac{e^x-e^{-x}}{e^x+e^{-x}}$ | $[-1, 1]$ | *Vanishing gradient* aux extrêmes | Anciennes couches cachées, RNN / LSTM |
| **ReLU**<br>$\max(0,x)$ | $[0,+\infty[$ | *Dying ReLU* : un neurone recevant toujours des valeurs négatives reste bloqué à 0 | Couches cachées standard (CNN, MLP) |
| **Softmax**<br>$\frac{e^{x_i}}{\sum e^{x_j}}$ | $[0, 1]$, somme = 1 | Sensible aux valeurs extrêmes (overflow numérique) | Couche de sortie en classification multi-classes |

!!! info "Variantes de ReLU"
- **Leaky ReLU** ($\max(0.01x, x)$) : conserve un très faible gradient pour $x < 0$ afin d'éviter la mort des neurones.
- **PReLU** : apprend dynamiquement ce coefficient de fuite.
- **ELU** : offre une courbe plus lisse pour les valeurs négatives.
- **GELU** : utilisé dans les architectures modernes de type Transformers (GPT, BERT).

!!! note "Softmax : des scores aux probabilités"
    Softmax transforme des scores bruts (logits) en un vecteur de probabilités dont la somme est égale à 1, en accentuant les écarts. 
    *Exemple :* Pour les scores `[2.0, 1.0, 0.5]`, softmax donne approximativement `[0.63, 0.23, 0.14]`.

---

## Le perceptron simple

Inventé par Frank Rosenblatt en 1957, c'est le modèle historique de neurone formel. Il combine un vecteur d'entrées, des poids, un biais et une fonction d'activation à seuil.

### Règle d'apprentissage
Pour chaque exemple d'entraînement $(x, y_{\text{vrai}})$ :

1. Calcul de la prédiction : 

$y_{\text{pred}} = f\left(\sum w_i x_i + w_0\right)$

2. Calcul de l'erreur commise :

 $e = y_{\text{vrai}} - y_{\text{pred}}$

3. Mise à jour des poids : 

$w_i \leftarrow w_i + \eta \cdot e \cdot x_i$

4. Mise à jour du biais :

 $w_0 \leftarrow w_0 + \eta \cdot e$

*(Où $\eta$ représente le **learning rate** ou taux d'apprentissage).*

!!! warning "Limites historiques du perceptron simple"
    - Il ne peut résoudre que des problèmes **linéairement séparables**.
    - Il échoue à apprendre des fonctions logiques basiques non-linéaires comme le **XOR** (car aucune droite unique ne peut séparer les classes du XOR).
    
    👉 **La solution :** Multiplier les couches de neurones pour former un Réseau Multicouches (MLP), capable de modéliser des frontières de décision complexes.