# TP3 : Prédiction de survie au naufrage du Titanic

## Objectif
Mener à bien une classification binaire complexe sur un jeu de données tabulaires réelles et hétérogènes (le Titanic) pour prédire si un passager a survécu ou non[cite: 90, 125].

---

## 1. Pipeline de prétraitement rigoureux
1. **Sélection des features** : Choix des colonnes pertinentes (`pclass`, `sex`, `age`, `sibsp`, `parch`, `fare`, `embarked`)[cite: 91, 127].
2. **Train/Test Split** : Séparation avec stratification[cite: 91, 127].
3. **Gestion des valeurs manquantes** : Imputation de l'âge par la *médiane* du train et du port d'embarquement par le *mode* du train[cite: 91, 127].
4. **Encodage catégoriel** : Utilisation de `pd.get_dummies` avec suppression de la première colonne pour éviter la colinéarité (`drop_first=True`)[cite: 91, 128].
5. **Normalisation numérique** : Standardisation des colonnes continues (`age`, `fare`)[cite: 91, 129].

## 2. Modélisation et Régularisation
* **Architecture** : Dense avec 27 neurones cachés (ReLU) et 1 neurone de sortie (Sigmoïde)[cite: 129, 130].
* **Early Stopping** : Utilisation d'un rappel (*callback*) pour interrompre l'entraînement lorsque la perte de validation ne s'améliore plus (patience de 10 époques) afin d'éviter le surapprentissage[cite: 91, 131].

---
> *Code source du notebook et rapport de classification (precision/recall) à venir.*