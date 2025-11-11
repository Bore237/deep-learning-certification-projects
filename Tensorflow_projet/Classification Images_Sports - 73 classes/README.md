# Classification Images_Sports - 73 classes

Résumé
-------
Ce projet est un classifieur d'images entraîné pour reconnaître 73 classes liées au sport. Il fait partie de la certification "OpenCV Deep Learning with TensorFlow" et montre la capacité à construire une pipeline d'entraînement et d'évaluation pour la classification multi-classes d'images.

Contenu du dossier
-------------------
- `project2-competition-starter-notebook-r4.ipynb` : Notebook principal montrant la préparation des données, la construction du modèle, l'entraînement et la génération d'une soumission.
- `submission+(1).csv` : Exemple de fichier de soumission.
- `log/` : TensorBoard logs pour l'entraînement et la validation.

Approche
--------
- Prétraitement : redimensionnement, normalisation, éventuellement augmentation (flip, rotation, etc.).
- Modèle : CNN basé sur une architecture moderne (ex : MobileNet / EfficientNet / ResNet selon le notebook), avec fine-tuning si un modèle pré-entraîné a été utilisé.
- Entraînement : optimiser avec Adam ou SGD, scheduler du learning rate, early stopping et checkpointing du meilleur modèle.

Résultats
---------
- Les métriques d'entraînement/validation sont loggées dans `log/` (TensorBoard). Le notebook contient le score final obtenu (voir cellule d'évaluation).

Comment exécuter
----------------
Pré-requis (recommandé)
- Python 3.8+ (ou 3.9)
- TensorFlow 2.x
- Pandas, NumPy, Matplotlib

Exemple (depuis l'environnement où se trouve le notebook) :
1. Ouvrir le notebook `project2-competition-starter-notebook-r4.ipynb` et exécuter les cellules dans l'ordre.
2. Pour lancer TensorBoard :

```powershell
# depuis le dossier racine du projet
tensorboard --logdir "Classification Images_Sports - 73 classes\log" --port 6006
``` 

Fichiers importants
- Notebook pour reproduire toutes les étapes.
- Logs TensorBoard pour analyser sur-apprentissage ou convergence.

Remarques / Améliorations possibles
- Ajouter une stratégie d'augmentation plus riche (MixUp, CutMix).
- Tester différentes architectures et comparer avec validation croisée.
- Déployer le modèle avec TensorFlow Serving ou convertir en TFLite pour infer sur mobile.

Contact
-------
Pour toute question concernant ce projet, voir le README général du dossier `Tensorflow_projet`.
