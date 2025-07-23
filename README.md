# 🦴 Pediatric Bone Fracture Detection with YOLOv8

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

🏥 **Détection automatique de fractures osseuses chez les enfants** à partir de radiographies médicales utilisant YOLOv8.

## 📋 Table des matières
- [🎯 Objectif](#-objectif)
- [🚀 Installation rapide](#-installation-rapide)
- [📁 Structure du projet](#-structure-du-projet)
- [🔧 Utilisation](#-utilisation)
- [📊 Dataset](#-dataset)
- [🧠 Modèle](#-modèle)
- [🖼️ Interface Web](#-interface-web)
- [📈 Évaluation](#-évaluation)
- [🤝 Contribution](#-contribution)

## 🎯 Objectif

Ce projet propose une **pipeline complète d'inférence** pour détecter automatiquement les fractures visibles sur les radiographies pédiatriques du poignet et de l'avant-bras. 

### Fonctionnalités principales :
- ✅ Détection de fractures avec YOLOv8
- ✅ Interface web interactive (Streamlit)
- ✅ Support image/vidéo/webcam
- ✅ Métriques d'évaluation complètes
- ✅ Pipeline prête pour la production

## 🚀 Installation rapide

### Prérequis
- Python 3.8+ 
- GPU recommandé (CUDA compatible)

### Installation
```bash
# Cloner le repository
git clone https://github.com/amdjadouxx/BONES_BREAK_DETECTION.git
cd BONES_BREAK_DETECTION

# Installer les dépendances
pip install -r requirements.txt

# Télécharger le dataset (optionnel)
python scripts/prepare_data.py --download

# Lancer l'interface web
streamlit run webapp/app.py
```

## 📁 Structure du projet

```
📦 BONES_BREAK_DETECTION/
├── 📁 data/                    # Gestion du dataset
│   ├── raw/                    # Données brutes
│   ├── processed/              # Données prétraitées
│   └── annotations/            # Annotations YOLO
├── 📁 models/                  # Modèles YOLOv8
│   ├── pretrained/             # Modèles pré-entraînés
│   └── checkpoints/            # Checkpoints d'entraînement
├── 📁 inference/               # Pipeline de prédiction
│   ├── predict.py              # Prédiction sur images
│   ├── video_inference.py      # Prédiction vidéo/webcam
│   └── batch_predict.py        # Prédictions par lots
├── 📁 webapp/                  # Interface Streamlit
│   ├── app.py                  # Application principale
│   └── components/             # Composants UI
├── 📁 notebooks/               # Analyse exploratoire
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_evaluation.ipynb
│   └── 03_results_analysis.ipynb
├── 📁 utils/                   # Fonctions utilitaires
│   ├── data_utils.py           # Prétraitement données
│   ├── model_utils.py          # Gestion modèles
│   ├── visualization.py        # Visualisations
│   └── metrics.py              # Métriques d'évaluation
├── 📁 scripts/                 # Scripts utilitaires
│   ├── prepare_data.py         # Préparation dataset
│   ├── convert_to_yolo.py      # Conversion format YOLO
│   └── evaluate_model.py       # Évaluation modèle
├── 📁 config/                  # Configurations
│   ├── model_config.yaml       # Config modèle
│   └── data_config.yaml        # Config dataset
├── 📄 requirements.txt         # Dépendances Python
├── 📄 setup.py                 # Installation package
└── 📄 README.md                # Documentation
```

## 🔧 Utilisation

### Prédiction sur une image
```python
from inference.predict import PediatricFractureDetector

# Initialiser le détecteur
detector = PediatricFractureDetector('models/best.pt')

# Prédiction
results = detector.predict('path/to/xray.jpg')
detector.save_results(results, 'output/')
```

### Prédiction via ligne de commande
```bash
# Image unique
python inference/predict.py --source path/to/image.jpg --output results/

# Dossier d'images
python inference/predict.py --source path/to/images/ --output results/

# Vidéo ou webcam
python inference/video_inference.py --source webcam --output results/
```

## 📊 Dataset

Ce projet utilise le **RSNA Pediatric Bone Age Dataset** et des datasets open-source de radiographies pédiatriques :

### Sources de données :
- 🔗 [RSNA Bone Age Challenge](https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/rsna-pediatric-bone-age-challenge-2017)

### Caractéristiques :
- **~12,000 radiographies** pédiatriques
- **Annotations** : bounding boxes des fractures
- **Métadonnées** : âge, sexe, localisation anatomique
- **Format** : DICOM → PNG/JPG + annotations YOLO

## 🧠 Modèle

### Architecture YOLOv8
- **Modèle base** : YOLOv8n/s/m/l/x (configurable)
- **Classes** : `fracture`, `no_fracture`
- **Input** : 640x640 pixels
- **Pré-traitement** : normalisation, augmentation

### Performance attendue :
- **Précision** : ~85-90%
- **Rappel** : ~80-85%
- **F1-Score** : ~82-87%
- **Temps d'inférence** : <100ms (GPU)

## 🖼️ Interface Web

Interface Streamlit intuitive permettant :
- 📤 Upload d'images médicales
- 🔍 Visualisation des prédictions
- 📊 Affichage des scores de confiance
- 💾 Téléchargement des résultats
- 📈 Historique des prédictions

## 📈 Évaluation

### Métriques disponibles :
- Précision, Rappel, F1-Score
- Matrice de confusion
- Courbes ROC/PR
- Heatmaps de confiance
- Analyse des erreurs

### Lancer l'évaluation :
```bash
python scripts/evaluate_model.py --model models/best.pt --data data/test/
```

### Développement local :
```bash
# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Installation en mode développement
pip install -e .

# Tests
python -m pytest tests/
```

---

## 📞 Contact

- **Auteur** : Amdjadouxx
- **Email** : amdjadahmodali974@gmail.com

## 📜 License

Ce projet est sous licence MIT. Voir [LICENSE](LICENSE) pour plus de détails.

---

⭐ **N'hésitez pas à starred ce projet si vous le trouvez utile !**
