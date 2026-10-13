# 3. Réseaux de neurones multicouches (MLP)

## Architecture

*   **Couche d'entrée** : reçoit les données brutes. Autant de neurones que de features, aucun calcul.
*   **Couches cachées** : transforment les données et extraient des features. Il peut y en avoir plusieurs (d'où « deep »). Chaque neurone est relié à tous ceux de la couche précédente (*fully connected*).
*   **Couche de sortie** : produit la prédiction. 1 neurone pour une régression ou une classification binaire, $k$ neurones pour $k$ classes.

## 🎥 Vidéo : Réseau de neurones profond (Machine Learnia)

<iframe width="560" height="315" src="https://www.youtube.com/embed/4NRLysodIwY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

---

## Propagation et Rétropropagation

### Forward propagation
Calculer la sortie du réseau à partir des entrées, couche après couche :

$z^{[l]} = W^{[l]} \cdot a^{[l-1]} + b^{[l]}$
puis $a^{[l]} = f(z^{[l]})$.

La sortie de la dernière couche est la prédiction $\hat{y}$.

### Rétropropagation (Backpropagation)
Elle calcule le gradient de la perte par rapport à chaque poids et chaque biais :

1. Calculer l'erreur en sortie, par exemple $\delta_{sortie} = (\hat{y} - y) \cdot f'(z_{sortie})$

2. La propager vers les couches précédentes avec la règle de la chaîne (chain rule).
3. Pour chaque couche, calculer la contribution de chaque poids à l'erreur finale.

### Mise à jour des paramètres
$W^{[l]} \leftarrow W^{[l]} - \eta \cdot \partial L/\partial W^{[l]}$

$b^{[l]} \leftarrow b^{[l]} - \eta \cdot \partial L/\partial b^{[l]}$

---

## Les Optimizers

Un optimizer ajuste les poids et les biais pour minimiser la fonction de coût.

| Optimizer | Principe | Avantages | Inconvénients | Usages courants |
| :--- | :--- | :--- | :--- | :--- |
| **SGD** | Descente de gradient par batch | Simple, pédagogique | Lent, sensible au LR | Cas simples, apprentissage |
| **Momentum** | Ajoute une inertie | Plus stable, évite la stagnation | Un paramètre de plus | Réseaux de base |
| **RMSProp** | LR adapté par paramètre | Bonne convergence | Parfois instable | RNN, séquences |
| **Adam** | Momentum + RMSProp | Rapide, efficace, robuste | Peut sur-ajuster | **Standard actuel** |

!!! tip "Comment choisir ?"
    *   **Pour débuter / en production** : Adam avec ses paramètres par défaut ($\eta=0.001$).
    *   **En cas de sur-apprentissage** : essaie SGD avec Momentum, ou AdamW.
    *   **Pour un RNN ou un LSTM** : RMSProp ou Adam.

---

## Le Learning Rate ($\eta$)

Le learning rate contrôle la taille du pas effectué lors de la mise à jour des poids. C'est le paramètre le plus important à régler.

*   **$\eta$ petit (ex: 0.0001)** : descente lente mais stable.
*   **$\eta$ grand (ex: 0.1)** : descente rapide au début, mais risque de diverger (la loss explose).
*   **Bon $\eta$ (0.001 à 0.01)** : équilibre entre rapidité et stabilité.

### Symptômes et solutions

| Symptôme | Cause probable | Solution |
| :--- | :--- | :--- |
| La loss diminue très lentement | $\eta$ trop petit | Augmenter $\eta$ ($\times2$ ou $\times10$) |
| La loss oscille fortement | $\eta$ trop grand | Réduire $\eta$ ($\div2$ ou $\div10$) |
| La loss explose (NaN) | $\eta$ beaucoup trop grand | Réduire drastiquement $\eta$ ($\div100$) |
| La loss stagne après un moment | $\eta$ devenu trop grand | Utiliser un schedule (decay) |