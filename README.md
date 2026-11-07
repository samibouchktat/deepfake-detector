# deepfake-detector
# Deepfake Detector

Projet en binôme de détection de **manipulations faciales dans les images et les vidéos**. L’application locale accepte un fichier, affiche une prédiction et un score de suspicion. Une visualisation pourra montrer les zones qui influencent le résultat image ou les passages vidéo les plus suspects.

**Présentation prévue le 17 novembre 2026.** Le dépôt est initialisé. Ce README décrit l’architecture et les fichiers à créer pendant le développement.

## Architecture commune

```text
Image → visage RGB ────────────────────────┐
                                           ├─ CNN partagé ─┬─ Tête image → score image
Vidéo → séquence ordonnée du même visage ───┘                └─ GRU + tête vidéo → score vidéo
```

Une seule classe `DeepfakeModel` contient le CNN partagé et les deux têtes. Le CNN traite une image à la fois avec **les mêmes poids** dans les deux modes. Dans la branche vidéo, la GRU reçoit les caractéristiques dans l’ordre des images.

**Entraînement from scratch.** Tous les poids du classificateur, y compris le CNN et la GRU, partent d’une initialisation aléatoire. Le script commun alterne les lots d’images et de vidéos. Chaque lot met à jour le CNN et la tête utilisée. L’entraînement produit un seul fichier `checkpoints/model.pt` qui permet de recharger les deux modes. L’équipe valide séparément le choix du détecteur de visages avec l’encadrant si le prétraitement utilise des poids externes.

## Répartition du travail

| Branche | Responsable | Livrable |
| --- | --- | --- |
| `feature/image` | Binôme image | Recadrage du visage, CNN partagé, tête image et préparation des exemples image |
| `feature/video` | Binôme vidéo | Manifeste des vidéos, extraction des séquences, tête vidéo avec GRU et agrégation des scores |
| `integration` | Ensemble | Assemblage du modèle, entraînement commun, évaluation et interface Streamlit |
| `main` | Ensemble | Version intégrée et vérifiée pour la démonstration |

### Contrat du modèle

| Élément | Entrée | Sortie |
| --- | --- | --- |
| CNN partagé | `[B, 3, 128, 128]` | `[B, 128]` |
| Tête image | `[B, 128]` | `[B]`, un logit par image |
| Tête vidéo | `[B, T, 128]` | `[B]`, un logit par séquence |
| `DeepfakeModel.forward_image` | `[B, 3, 128, 128]` | `[B]` |
| `DeepfakeModel.forward_video` | `[B, T, 3, 128, 128]` | `[B]` |

`B` désigne la taille du lot. Pour la première version, `T = 16` et la vidéo est échantillonnée à **8 images par seconde**. Les entrées du CNN sont des tenseurs RGB `float32` dans `[0, 1]`, avec les canaux en premier. Le recadrage et la mise à l’échelle sont communs à l’entraînement et à la prédiction. Chaque tête renvoie un logit brut pour `BCEWithLogitsLoss`.

Pour prédire sur une vidéo, analysez plusieurs fenêtres du visage principal, moyennez leurs logits, puis appliquez la sigmoïde. Pour une image, appliquez la sigmoïde à son logit. Les deux fonctions `predict_image(path)` et `predict_video(path)` partagent ces champs pour l’interface.

| Champ | Convention |
| --- | --- |
| `status` | `ok` ou `inconclusive` |
| `label` | `real` ou `fake` si `status` vaut `ok` |
| `fake_score` | Score entre 0 et 1 si `status` vaut `ok` |
| `evidence` | Facultatif. Carte Grad-CAM pour l’image ou fenêtres horodatées pour la vidéo |

Utilisez `inconclusive` et laissez `label` et `fake_score` vides lorsqu’aucun visage ou aucune fenêtre n’est exploitable. Choisissez un seuil par mode sur la validation. Affichez le résultat comme un **score de suspicion**.

### Structure prévue

```text
deepfake-detector/
├── README.md
├── .gitignore
├── common/
│   └── preprocessing.py  # Recadrage, RGB et conversion en tenseur
├── data/
│   ├── manifest.csv      # Découpage commun et chemins relatifs
│   ├── prepare_images.py # Extraction des visages pour la partie image
│   ├── prepare_videos.py # Extraction des séquences pour la partie vidéo
│   └── raw/              # Données locales, ignorées par Git
├── model/
│   ├── encoder.py        # CNN partagé
│   ├── image_head.py
│   ├── video_head.py
│   └── detector.py       # DeepfakeModel et ses deux modes
├── train.py              # Entraînement commun
├── app.py                # Interface Streamlit
└── checkpoints/         # Poids et paramètres, ignorés par Git
```

## Installation locale sous Windows

Utilisez Python 3.11. Après avoir accepté l’invitation GitHub, le binôme image clone directement sa branche. Le binôme vidéo peut ouvrir sa branche dans son clone existant avec `git switch feature/video`.

```powershell
git clone --branch feature/image https://github.com/samibouchktat/deepfake-detector.git
cd deepfake-detector
python -m venv .venv
```

Choisissez une seule commande PyTorch selon la machine. La version CUDA 12.6 est celle prévue pour l’AORUS. La version CPU suffit pour développer et vérifier le modèle sur de petits lots. [Versions et commandes PyTorch](https://pytorch.org/get-started/previous-versions/)

```powershell
# AORUS avec GPU NVIDIA
.\.venv\Scripts\python.exe -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu126

# Machine utilisant le CPU
.\.venv\Scripts\python.exe -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
```

```powershell
.\.venv\Scripts\python.exe -m pip install opencv-python streamlit scikit-learn
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print('CUDA disponible :', torch.cuda.is_available())"
```

Sur l’AORUS, vérifiez aussi qu’une opération s’exécute sur le GPU.

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.ones(1, device='cuda'))"
```

Une fois `app.py` créé et le checkpoint installé dans `checkpoints/`, lancez la démo avec `.\.venv\Scripts\python.exe -m streamlit run app.py`.

## Données et évaluation

1. **Démarrage avec SDFVD.** Ce jeu contient 53 vidéos originales et 53 vidéos manipulées. Utilisez-le pour préparer les deux pipelines et la première démo. Si l’accès bloque ou si vous ne pouvez pas relier les vidéos manipulées à leurs sources, envisagez l’échantillon DFDC et son `metadata.json`. Décidez ensemble du jeu retenu. [SDFVD sur Mendeley Data](https://data.mendeley.com/datasets/bcmkfgct2s/1), [échantillon DFDC sur Kaggle](https://www.kaggle.com/competitions/deepfake-detection-challenge/data?select=sample_submission.csv)
2. **Un manifeste versionné.** Utilisez les colonnes `video_id,source_id,label,split,path`, avec `0 = real` et `1 = fake`. Le chemin est relatif au dépôt. Si une manipulation dépend de plusieurs vidéos sources identifiées, gardez toutes les vidéos liées dans le même groupe avant le découpage.
3. **Découpage avant extraction.** Figez `train`, `val` et `test` par groupe de sources. Une vidéo originale, ses variantes et leurs images ou séquences restent dans le même ensemble. Le binôme image utilise des images extraites. Le binôme vidéo utilise des séquences issues des mêmes groupes.
4. **Évaluation.** Choisissez les seuils sur `val`. Sur `test`, mesurez le ROC-AUC, le F1 de la classe `fake` et la matrice de confusion, par image ou par vidéo selon le mode. Indiquez combien de fichiers ont donné un résultat `inconclusive`. Un résultat sur SDFVD décrit le comportement du prototype sur ce jeu.

## Collaboration GitHub

```text
feature/image ──┐
                ├── integration ── validation à deux ── main
feature/video ──┘
```

1. Travaillez sur votre branche et faites des commits ciblés. Signalez à l’autre tout changement du prétraitement, du manifeste ou du contrat du modèle avant de le fusionner.
2. Ouvrez chacun une pull request avec `integration` comme branche de destination. Ajoutez une commande ou un exemple qui permet à l’autre de vérifier votre partie.
3. Sur `integration`, assemblez les deux modes, lancez un entraînement de vérification, rechargez le même `model.pt` et essayez une image et une vidéo. Vérifiez aussi le cas sans visage exploitable et les exemples réels et manipulés du jeu de test.
4. Ouvrez ensuite une pull request `integration` → `main`. Partagez le checkpoint à part avec sa configuration de prétraitement, ses seuils et la référence du commit validé. `.venv/`, `data/raw/` et `checkpoints/` restent locaux.

**Repères avant la présentation** · Visez un premier passage image et vidéo dans l’interface le 11 novembre, figez la version de démonstration le 15 et répétez sur l’AORUS le 16. Ajoutez les visualisations si les deux modes sont déjà vérifiés.

## Démarrer avec Codex

Ouvrez le dossier cloné dans VS Code ou dans Codex. Copiez le message correspondant à votre branche pour donner à Codex le contexte et une première tâche concrète.

### Message pour le binôme image

> Lis le `README.md` à la racine et vérifie que je suis sur `feature/image`. Nous construisons un seul modèle PyTorch de détection de deepfakes faciaux, entraîné depuis des poids aléatoires. Je suis responsable du CNN partagé et de la tête image. Commence par implémenter `model/encoder.py` et `model/image_head.py` avec le contrat du README. Vérifie sur un petit lot de tenseurs fictifs que le CNN produit `[B, 128]` et que la tête produit `[B]`. Utilise le CPU pour cette vérification. Garde le contrat compatible avec la branche vidéo et indique les fichiers modifiés et la commande de vérification.

### Message pour le binôme vidéo

> Lis le `README.md` à la racine et vérifie que je suis sur `feature/video`. Nous construisons un seul modèle PyTorch de détection de deepfakes faciaux, avec un CNN partagé. Je suis responsable de la tête vidéo. Commence par implémenter `model/video_head.py` avec une GRU qui transforme des séquences `[B, 16, 128]` en logits `[B]`. Vérifie-la avec des vecteurs fictifs, puis indique les fichiers modifiés et la commande de vérification. Nous brancherons le CNN du binôme sur `integration`.
