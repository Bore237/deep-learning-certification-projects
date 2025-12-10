# MedSam - Medical Segmentation with Segment Anything Model

## 📋 Description

MedSam est un projet de segmentation médicale basé sur le modèle **Segment Anything (SAM)** adapté pour les images médicales. Ce projet démontre l'application de techniques avancées de segmentation aux images médicales 3D.

## 🎯 Objectifs

- Utiliser le modèle pré-entraîné SAM pour la segmentation d'images médicales
- Adapter les prompts textuels pour identifier des structures anatomiques spécifiques
- Évaluer les performances de segmentation sur des données médicales (FLARE22, BraTS)
- Optimiser les hyperparamètres pour la segmentation précise

## 📚 Concepts de Segmentation Maîtrisés

### Vision par Ordinateur
- **Segmentation sémantique** : Classification au niveau des pixels
- **Segmentation d'instance** : Identification d'objets individuels
- **Segmentation interactive** : Utilisation de prompts (points, boîtes, texte)

### Architectures et Modèles
- **Vision Transformer (ViT)** : Utilisation de transformer pour la vision
- **Modèle SAM** : Adaptation d'un modèle fondation pour la segmentation médicale
- **Transfer Learning** : Fine-tuning sur des données médicales spécifiques

### Traitement d'Images Médicales
- **Prétraitement MRI** : Normalisation et augmentation de contraste
- **Formats NIFTI** : Gestion des images 3D médicales
- **Volumes 3D** : Segmentation par slices et reconstruction

## 🗂️ Structure du Projet

```
MedSam/
├── medsam.ipynb              # Notebook principal avec expériences complètes
├── medsam - v0.ipynb         # Version initiale
├── medsam - v1.ipynb         # Améliorations et optimisations
├── model/
│   ├── medsam_vit_b.pth      # Poids du modèle SAM (ViT-B)
│   └── medsam_text_prompt_flare22.pth  # Modèle fine-tuné texte
└── utils/
    ├── medsam_data.py        # Chargement et traitement des données
    ├── medsam_eval.py        # Métriques d'évaluation (Dice, IOU)
    ├── mri_preprocessing.py  # Pipeline de prétraitement MRI
    └── liveplot.py           # Visualisation en temps réel
```

## 🔧 Technologies Utilisées

- **PyTorch** : Framework deep learning
- **MONAI** / **Nibabel** : Manipulation d'images médicales
- **Segment Anything** : Modèle de segmentation fondation
- **OpenCV** : Traitement d'images
- **Scikit-image** : Outils de segmentation complémentaires

## 📊 Méthodologie

1. **Chargement des données** : Images NIFTI du dataset FLARE22/BraTS
2. **Prétraitement** : Normalisation, redimensionnement, augmentation
3. **Inférence** : Utilisation de SAM avec prompts textuels/spatiaux
4. **Post-traitement** : Nettoyage morphologique, remplissage des trous
5. **Évaluation** : Calcul des métriques (Dice, Hausdorff, IOU)
6. **Visualisation** : Comparaison prédiction vs ground truth

## 💡 Apprentissages Clés

✅ **Adaptation de modèles fondations** aux domaines spécifiques  
✅ **Segmentation interactive** vs approches traditionnelles  
✅ **Gestion de données 3D** et volumes médicaux  
✅ **Fine-tuning efficace** pour améliorer les performances  
✅ **Métriques adaptées** au contexte médical (sensibilité/spécificité)

## 🚀 Utilisation

```python
# Charger un modèle pré-entraîné
from medsam import MedSAM

model = MedSAM()

# Inférence avec prompts textuels
prediction = model.predict(image, prompt="kidney tumor")

# Évaluation
dice_score = calculate_dice(prediction, ground_truth)
```

## 📈 Résultats

- Segmentation précise avec prompts textuels
- Adaptation rapide à de nouveaux organes/pathologies
- Performance compétitive sur benchmarks médicaux

---

**Auteur** : Segmentation Project  
**Date** : Décembre 2025
