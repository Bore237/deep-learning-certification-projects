# Segmentation — FloodNet dataset (10 masks)

Résumé
-------
Projet de segmentation sémantique utilisant le jeu de données FloodNet (10 classes de masques). L'objectif est de segmenter différentes zones liées aux inondations (eau, bâtiments, routes, etc.). Ce projet illustre la préparation des masques, la définition d'un modèle de segmentation (U-Net / DeepLab / autre), et l'évaluation des métriques de segmentation.

Contenu du dossier
-------------------
- `model.keras` : modèle entraîné (format Keras).
- `Train/` et `Test/` : logs TensorBoard et/ou éventuellement sous-ensembles utilisés.

Approche
--------
- Prétraitement des images et masques (redimension, encodage des classes dans des canaux ou une carte d'entiers).
- Modèle : architecture de segmentation (U-Net typique ou variations basées sur encoders pré-entraînés).
- Loss : combinaison de Dice Loss et Cross-Entropy selon l'équilibre des classes.

Résultats
---------
- Modèle sauvegardé sous `model.keras`. Utiliser le notebook associé (s'il existe) pour voir l'entraînement et les métriques.

Comment exécuter
----------------
Prérequis : Python 3.8+, TensorFlow 2.x, opencv-python, scikit-image, albumentations (optionnel pour augmentation).
- Charger `model.keras` :

```python
from tensorflow import keras
model = keras.models.load_model('segmentation  FloodNet dataset - 10 masks/model.keras')
```

- Préparer une image (mêmes prétraitements que durant l'entraînement) puis appeler `model.predict`.

Améliorations possibles
- Améliorer le prétraitement des masques (ex : gestion des classes non-mutuellement exclusives si nécessaire).
- Utiliser backbone plus puissant pour encoder (ResNet/EfficientNet) et fine-tuning.

Notes
-----
- Vérifier que la correspondance classes <-> indices est documentée dans le notebook original ou dans les métadonnées du dataset.
- Pour la production, exporter en SavedModel et écrire un petit service d'inférence.
