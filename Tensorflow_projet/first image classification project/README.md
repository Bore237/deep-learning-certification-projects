# First image classification project

Résumé
-------
Projet d'introduction à la classification d'images. L'objectif est de construire une pipeline complète : nettoyage des données, préparation, définition d'un modèle CNN, entraînement et évaluation.

Contenu du dossier
-------------------
- `data-and-pipeline-check-starter-notebook-r4.ipynb` : Notebook de préparation et d'analyse exploratoire.
- `training-from-scratch-starter-notebook-r4+(1).ipynb` : Notebook montrant l'entraînement depuis zéro.
- `sample_submission.csv`, `submission.csv` : exemples de sorties.

Approche
--------
- Étapes : EDA, nettoyage, split train/validation, pipeline TF Data ou generators, définition du modèle (CNN simple ou pré-entraîné), entraînement et métriques.
- Entraînement depuis zéro possible (training-from-scratch notebook) pour comprendre toutes les étapes.

Comment exécuter
----------------
Prérequis : Python 3.8+, TensorFlow 2.x, scikit-learn, pandas.
- Ouvrir les notebooks et exécuter les cellules.
- Les notebooks contiennent des cellules pour charger un jeu de données local ou simuler un petit jeu d'exemples pour test rapide.

Améliorations possibles
- Introduction du fine-tuning avec un backbone pré-entraîné pour améliorer la précision.
- Ajout de callbacks avancés (ModelCheckpoint, ReduceLROnPlateau, EarlyStopping).

Notes
-----
Ce dossier sert de démonstration pédagogique : il illustre la chaîne complète depuis les données brutes jusqu'au modèle entraîné.
