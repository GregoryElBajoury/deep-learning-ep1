# 2. Le neurone artificiel et le perceptron

## Anatomie d'un neurone

Un neurone artificiel est composé de plusieurs éléments :

*   **Entrées ($x_1...x_n$)** : les données brutes ou les signaux venus d'autres neurones.
*   **Poids ($w_1...w_n$)** : l'importance de chaque entrée. Un poids élevé donne beaucoup d'influence.
*   **Biais ($b$)** : une constante qui décale l'activation, comme un seuil personnalisé.

Le calcul s'effectue en deux étapes :
1.  **Somme pondérée $z$** : $z = w_1x_1 + w_2x_2 + ... + w_nx_n + b = \Sigma(w_ix_i) + b$
2.  **Fonction d'activation $f$** : donne la sortie $y = f(z)$

## 🎥 Vidéo : Le Perceptron (Machine Learnia)

<iframe width="560" height="315" src="https://www.youtube.com/embed/VlMm4VZ6lk4" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

---

## Les fonctions d'activation

Elles introduisent la **non-linéarité**, ce qui permet au réseau de modéliser des relations complexes.

| Fonction | Sortie | Problèmes principaux | Utilisation typique |
| :--- | :--- | :--- | :--- |
| **Sigmoïde**<br>$1/(1+e^{-x})$ | $[0, 1]$ | Vanishing gradient, non centrée en 0, exponentielle coûteuse | Sortie d'une classification binaire (probabilité de la classe positive) |
| **Tanh**<br>$(e^x-e^{-x})/(e^x+e^{-x})$ | $[-1,1]$ | Vanishing gradient aux extrêmes | Anciennes couches cachées, certains RNN et LSTM |
| **ReLU**<br>$\max(0,x)$ | $[0,+\infty[$ | Dying ReLU: un neurone qui ne reçoit que des valeurs négatives reste bloqué à 0 | Couches cachées (standard actuel) des CNN et des MLP |
| **Softmax**<br>$e^x / \Sigma e^x$ | $[0, 1]$,<br>somme = 1 | Sensible aux valeurs extrêmes (overflow numérique) | Sortie d'une classification multi-classes, attention des Transformers |

!!! info "Variantes de ReLU"
    *   **Leaky ReLU** $\max(0.01x, x)$ : garde un petit gradient pour $x<0$.
    *   **PReLU** : apprend ce coefficient.
    *   **ELU** : lisse pour les valeurs négatives.
    *   **GELU** : sert dans les Transformers (GPT, BERT).

!!! note "Softmax : des scores aux probabilités"
    Softmax transforme des scores quelconques en probabilités qui somment à 1, et accentue les écarts : le score le plus haut prend une part bien plus grande.
    Exemple : Pour les scores `[2.0, 1.0, 0.5]`, softmax donne `[0.63, 0.23, 0.14]`.

---

## Le perceptron simple

C'est le plus petit réseau possible : un seul neurone, avec $n$ entrées, $n$ poids, un biais, une fonction d'activation (en général une fonction échelon ou la sigmoïde) et une sortie.

On calcule $z = \Sigma(w_ix_i) + w_0$, puis $y = f(z)$; si $y$ dépasse un seuil, c'est la classe 1, sinon la classe 0.

### Règle d'apprentissage

Pour chaque exemple $(x, y_{vrai})$ :

1.  $y_{predit} = f(\Sigma w_ix_i + w_0)$
2.  erreur $e = y_{vrai} - y_{predit}$
3.  $w_i \leftarrow w_i + \eta \cdot e \cdot x_i$
4.  $w_0 \leftarrow w_0 + \eta \cdot e$

*$\eta$ (êta) est le learning rate.*

!!! warning "Limites du perceptron simple"
    *   Il ne résout que les problèmes **linéairement séparables**.
    *   Il ne peut pas apprendre la porte logique **XOR**.
    *   Sa puissance de représentation est limitée.
    
    👉 **La solution :** empiler plusieurs couches de neurones (MLP).