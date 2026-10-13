# 4. Les architectures spécialisées : CNN et RNN

> Plongée au cœur des réseaux de neurones spécialisés : comment traiter la vision par ordinateur avec les convolutions (CNN) et analyser les séquences textuelles ou temporelles avec la récurrence (RNN & LSTM).

!!! abstract "Objectifs d'apprentissage"
    À la fin de ce chapitre, tu sauras :
- Expliquer en détail le rôle de chaque couche d'un réseau convolutif (`Conv2D`, `MaxPooling2D`, `Flatten`, `Dropout`).
- Comprendre pourquoi les réseaux fully-connected (MLP) s'avèrent inefficaces pour traiter les images.
- Appréhender le fonctionnement séquentiel des réseaux récurrents (`SimpleRNN`) et leurs évolutions à portes (`LSTM`).
- Maîtriser l'utilité des couches d'incorporation (`Embedding`) pour le traitement du langage naturel (NLP).

---

## 1. Vision par ordinateur : Les Réseaux Convolutifs (CNN)

### Pourquoi les MLP échouent sur les images ?
Imaginons une image modeste en niveaux de gris de $28 \times 28$ pixels (comme le dataset MNIST). Cela représente déjà $784$ pixels d'entrée. Si la première couche cachée d'un MLP comporte par exemple $512$ neurones, le nombre de poids à apprendre rien que pour cette première connexion est de $784 \times 512 \approx 400\,000$ paramètres ! 
Pour une image couleur moderne de $256 \times 256$ pixels, le nombre de paramètres explose littéralement, menant inévitablement à un surapprentissage massif (*overfitting*) et à une perte totale de la structure spatiale (le fait qu'un pixel soit voisin d'un autre).

La solution réside dans les **CNN (Convolutional Neural Networks)**, qui s'inspirent directement du cortex visuel humain en extrayant des motifs locaux grâce au partage de poids.

---

### Anatomie d'un bloc convolutif

Un CNN alterne principalement deux types de couches pour l'extraction des caractéristiques (*Feature Extraction*), suivies de couches denses pour la classification.

| Composant du CNN | Rôle principal | Fonctionnement technique |
| :--- | :--- | :--- |
| **`Conv2D`** (Convolution) | Détecte des motifs visuels locaux (bords, lignes, textures, formes géométriques). | Glisse un filtre (noyau, ex: $3 \times 3$) sur l'image en réalisant un produit scalaire pour générer une *feature map* (carte de caractéristiques). |
| **`MaxPooling2D`** (Sous-échantillonnage) | Réduit la dimension spatiale de l'image, diminue le coût de calcul et offre une invariance aux légères translations. | Conserve uniquement la valeur maximale (ou moyenne) dans une fenêtre glissante (ex: $2 \times 2$). |
| **`Flatten`** (Aplatissement) | Fait la transition entre l'extraction spatiale et la classification. | Transforme les cartes de caractéristiques 2D en un vecteur unidimensionnel 1D. |
| **`Dense`** (Totalement connecté) | Combine les caractéristiques de haut niveau pour attribuer un score aux classes finales. | Connexions classiques de neurones (MLP) avec une fonction d'activation (`ReLU` puis `Softmax`). |
| **`Dropout`** (Régularisation) | Lutte activement contre le surapprentissage. | Désactive aléatoirement un pourcentage de neurones (ex: 50%) à chaque étape d'entraînement. |

---

### La hiérarchie des couches dans un CNN
Plus on progresse vers les couches profondes du réseau, plus la complexité des motifs détectés augmente :

1. **Couches basses** : Détection des contours, des orientations et des contrastes simples.

2. **Couches intermédiaires** : Combinaison des contours pour identifier des textures et des formes géométriques basiques (cercles, coins).

3. **Couches hautes** : Reconnaissance de parties d'objets spécifiques (yeux, roues, anses de tasses).

4. **Sortie** : Classification globale de l'objet ou de l'image (ex: chiffre $5$, vêtement de type *Pantalon*).

---

## 2. Traitement séquentiel : Les Réseaux Récurrents (RNN & LSTM)

### Le défi des données séquentielles
Les données textuelles, les séries temporelles financières ou les flux audio possèdent une dimension temporelle ou ordonnée : **l'ordre des mots ou des événements change complètement le sens**. Un réseau standard (MLP ou CNN) traite chaque entrée de manière indépendante et ne possède aucune « mémoire » du passé.

Les **RNN (Recurrent Neural Networks)** introduisent une boucle de rétroaction interne : à chaque pas de temps $t$, le neurone reçoit à la fois l'élément actuel $x_t$ et son propre état caché précédent $h_{t-1}$.

---

### Comparatif : SimpleRNN vs LSTM

| Architecture | Principe de fonctionnement | Limites majeures | Cas d'usage typique |
| :--- | :--- | :--- | :--- |
| **`SimpleRNN`** | Réactualise son état caché de façon récursive à chaque mot. | Souffre cruellement du *vanishing gradient* sur les phrases longues : il « oublie » le début de la phrase. | Mini-projets pédagogiques, séquences très courtes. |
| **`LSTM`** *(Long Short-Term Memory)* | Introduit un « tapis roulant mémoriel » régulé par trois portes (*oubli*, *entrée*, *sortie*). | Modèle plus lourd, nécessite davantage de temps de calcul et de données. | Génération de texte, traduction automatique, séries temporelles complexes. |

---

### Le rôle de la couche `Embedding`
Avant d'entrer dans un réseau récurrent, les mots textuels doivent être convertis en vecteurs numériques.

* Un encodage brutal de type *one-hot* produit des vecteurs immenses et creux, incapables de capturer la sémantique (par exemple, le fait que « chat » et « félins » soient proches).

* La couche **`Embedding`** agit comme une table de correspondance apprenable : elle projette chaque mot du vocabulaire dans un espace vectoriel continu de dimension fixe (ex: 16 ou 64 dimensions), où des mots au sens proche se retrouvent géométriquement rapprochés.

---

## 3. Bonnes pratiques MLOps : Éviter le *Data Leakage*

Quel que soit le modèle choisi (MLP, CNN ou RNN), une règle d'or absolue s'applique en Data Engineering et en Deep Learning : **l'isolation stricte du jeu de test**.

!!! warning "Rappel critique sur la fuite de données (*Data Leakage*)"
    - **La normalisation (`StandardScaler` / division par 255)** : Les paramètres statistiques (moyenne, écart-type) doivent **toujours** être calculés exclusivement sur le jeu d'entraînement (`X_train`), puis appliqués tels quels au jeu de test (`X_test`). 
    - Si l'on calcule les métriques sur l'ensemble global des données avant la séparation, le modèle « devine » indirectement des informations sur le test, faussant totalement l'évaluation finale des performances de généralisation.