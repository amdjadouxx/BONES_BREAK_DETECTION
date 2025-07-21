"""
Utilitaires pour la gestion et le chargement des modèles YOLOv8.
"""

import torch
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from ultralytics import YOLO
from loguru import logger
import numpy as np


class ModelManager:
    """Gestionnaire pour les modèles YOLOv8."""

    def __init__(self, config_path: str = "config/model_config.yaml"):
        """
        Initialise le gestionnaire de modèles.

        Args:
            config_path: Chemin vers le fichier de configuration
        """
        self.config = self.load_config(config_path)
        self.model = None
        self.device = self.get_device()

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Charge la configuration du modèle."""
        try:
            with open(config_path, "r", encoding="utf-8") as file:
                config = yaml.safe_load(file)
            logger.info(f"Configuration modèle chargée depuis {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(
                f"Fichier de config non trouvé: {config_path}, utilisation des valeurs par défaut"
            )
            return self.get_default_config()

    def get_default_config(self) -> Dict[str, Any]:
        """Configuration par défaut."""
        return {
            "model": {
                "name": "yolov8n",
                "num_classes": 1,
                "class_names": {0: "fracture"},
                "confidence_threshold": 0.25,
                "iou_threshold": 0.45,
                "input_size": 640,
            },
            "inference": {"device": "auto", "half_precision": False},
        }

    def get_device(self) -> str:
        """Détermine le meilleur device disponible."""
        device_config = self.config.get("inference", {}).get("device", "auto")

        if device_config == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"GPU CUDA détecté: {torch.cuda.get_device_name(0)}")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
                logger.info("GPU Apple Silicon (MPS) détecté")
            else:
                device = "cpu"
                logger.info("Utilisation du CPU")
        else:
            device = device_config
            logger.info(f"Device forcé: {device}")

        return device

    def load_model(self, model_path: str = None) -> YOLO:
        """
        Charge un modèle YOLOv8.

        Args:
            model_path: Chemin vers le modèle (si None, utilise le modèle pré-entraîné)

        Returns:
            Instance du modèle YOLO
        """
        try:
            if model_path and Path(model_path).exists():
                # Charger modèle personnalisé
                self.model = YOLO(model_path)
                logger.info(f"Modèle personnalisé chargé: {model_path}")
            else:
                # Charger modèle pré-entraîné
                model_name = self.config["model"]["name"]
                self.model = YOLO(f"{model_name}.pt")
                logger.info(f"Modèle pré-entraîné chargé: {model_name}")

            # Configurer le device
            if self.device != "cpu":
                self.model.to(self.device)

            return self.model

        except Exception as e:
            logger.error(f"Erreur chargement modèle: {e}")
            raise

    def predict(
        self,
        source: str,
        conf: float = None,
        iou: float = None,
        save: bool = False,
        save_dir: str = "runs/detect",
    ) -> List[Any]:
        """
        Effectue une prédiction avec le modèle.

        Args:
            source: Source (image, vidéo, dossier)
            conf: Seuil de confiance
            iou: Seuil IoU pour NMS
            save: Sauvegarder les résultats
            save_dir: Répertoire de sauvegarde

        Returns:
            Résultats de détection
        """
        if self.model is None:
            raise ValueError("Aucun modèle chargé. Utilisez load_model() d'abord.")

        # Utiliser les seuils de configuration si non spécifiés
        if conf is None:
            conf = self.config["model"]["confidence_threshold"]
        if iou is None:
            iou = self.config["model"]["iou_threshold"]

        try:
            results = self.model(
                source=source,
                conf=conf,
                iou=iou,
                save=save,
                project=save_dir,
                device=self.device,
                half=self.config.get("inference", {}).get("half_precision", False),
            )

            logger.info(f"Prédiction effectuée sur: {source}")
            return results

        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {e}")
            raise

    def train(self, data_config: str, epochs: int = None, **kwargs) -> Any:
        """
        Entraîne le modèle.

        Args:
            data_config: Chemin vers le fichier de configuration des données
            epochs: Nombre d'époques
            **kwargs: Arguments supplémentaires pour l'entraînement

        Returns:
            Résultats de l'entraînement
        """
        if self.model is None:
            raise ValueError("Aucun modèle chargé. Utilisez load_model() d'abord.")

        # Paramètres par défaut depuis la config
        train_config = self.config.get("training", {})

        training_args = {
            "data": data_config,
            "epochs": epochs or train_config.get("epochs", 100),
            "batch": train_config.get("batch_size", 16),
            "lr0": train_config.get("learning_rate", 0.001),
            "device": self.device,
            "project": "runs/train",
            "name": "fracture_detection",
            **kwargs,
        }

        try:
            logger.info(
                f"Début de l'entraînement avec {training_args['epochs']} époques"
            )
            results = self.model.train(**training_args)
            logger.info("Entraînement terminé avec succès")
            return results

        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement: {e}")
            raise

    def validate(self, data_config: str = None) -> Dict[str, float]:
        """
        Valide le modèle sur un dataset de test.

        Args:
            data_config: Chemin vers la configuration des données

        Returns:
            Métriques de validation
        """
        if self.model is None:
            raise ValueError("Aucun modèle chargé. Utilisez load_model() d'abord.")

        try:
            if data_config:
                results = self.model.val(data=data_config, device=self.device)
            else:
                results = self.model.val(device=self.device)

            # Extraire les métriques principales
            metrics = {
                "precision": (
                    float(results.box.mp) if hasattr(results.box, "mp") else 0.0
                ),
                "recall": float(results.box.mr) if hasattr(results.box, "mr") else 0.0,
                "mAP50": (
                    float(results.box.map50) if hasattr(results.box, "map50") else 0.0
                ),
                "mAP50-95": (
                    float(results.box.map) if hasattr(results.box, "map") else 0.0
                ),
            }

            logger.info(f"Validation terminée - mAP50: {metrics['mAP50']:.3f}")
            return metrics

        except Exception as e:
            logger.error(f"Erreur lors de la validation: {e}")
            raise

    def export(self, format: str = "onnx", **kwargs) -> str:
        """
        Exporte le modèle dans un format spécifique.

        Args:
            format: Format d'export ('onnx', 'tensorrt', 'coreml', etc.)
            **kwargs: Arguments supplémentaires pour l'export

        Returns:
            Chemin du modèle exporté
        """
        if self.model is None:
            raise ValueError("Aucun modèle chargé. Utilisez load_model() d'abord.")

        try:
            export_path = self.model.export(format=format, **kwargs)
            logger.info(f"Modèle exporté en {format}: {export_path}")
            return export_path

        except Exception as e:
            logger.error(f"Erreur lors de l'export: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """
        Retourne les informations du modèle chargé.

        Returns:
            Dictionnaire avec les informations du modèle
        """
        if self.model is None:
            return {"status": "Aucun modèle chargé"}

        info = {
            "model_name": self.config["model"]["name"],
            "num_classes": self.config["model"]["num_classes"],
            "class_names": self.config["model"]["class_names"],
            "device": self.device,
            "input_size": self.config["model"]["input_size"],
            "confidence_threshold": self.config["model"]["confidence_threshold"],
            "iou_threshold": self.config["model"]["iou_threshold"],
        }

        # Ajouter info sur les paramètres si disponible
        try:
            total_params = sum(p.numel() for p in self.model.model.parameters())
            trainable_params = sum(
                p.numel() for p in self.model.model.parameters() if p.requires_grad
            )
            info.update(
                {
                    "total_parameters": total_params,
                    "trainable_parameters": trainable_params,
                }
            )
        except:
            pass

        return info


class PredictionPostProcessor:
    """Post-traitement des prédictions YOLOv8."""

    def __init__(self, class_names: Dict[int, str] = None):
        self.class_names = class_names or {0: "fracture"}

    def process_results(self, results: List[Any]) -> List[Dict[str, Any]]:
        """
        Traite les résultats bruts de YOLOv8.

        Args:
            results: Résultats bruts de YOLOv8

        Returns:
            Liste des détections formatées
        """
        processed_results = []

        for result in results:
            detections = []

            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)

                for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                    detection = {
                        "bbox": {
                            "x1": float(box[0]),
                            "y1": float(box[1]),
                            "x2": float(box[2]),
                            "y2": float(box[3]),
                            "width": float(box[2] - box[0]),
                            "height": float(box[3] - box[1]),
                            "center_x": float((box[0] + box[2]) / 2),
                            "center_y": float((box[1] + box[3]) / 2),
                        },
                        "confidence": float(conf),
                        "class_id": int(cls),
                        "class_name": self.class_names.get(cls, f"class_{cls}"),
                        "detection_id": i,
                    }
                    detections.append(detection)

            processed_results.append(
                {
                    "image_path": result.path if hasattr(result, "path") else None,
                    "image_shape": (
                        result.orig_shape if hasattr(result, "orig_shape") else None
                    ),
                    "detections": detections,
                    "detection_count": len(detections),
                }
            )

        return processed_results

    def filter_by_confidence(
        self, detections: List[Dict[str, Any]], min_confidence: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Filtre les détections par seuil de confiance."""
        return [det for det in detections if det["confidence"] >= min_confidence]

    def filter_by_area(
        self,
        detections: List[Dict[str, Any]],
        min_area: int = 100,
        max_area: int = None,
    ) -> List[Dict[str, Any]]:
        """Filtre les détections par taille de bounding box."""
        filtered = []
        for det in detections:
            area = det["bbox"]["width"] * det["bbox"]["height"]
            if area >= min_area and (max_area is None or area <= max_area):
                filtered.append(det)
        return filtered


def load_pretrained_model(model_name: str = "yolov8n") -> YOLO:
    """
    Charge rapidement un modèle YOLOv8 pré-entraîné.

    Args:
        model_name: Nom du modèle (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)

    Returns:
        Instance du modèle YOLO
    """
    try:
        model = YOLO(f"{model_name}.pt")
        logger.info(f"Modèle {model_name} chargé avec succès")
        return model
    except Exception as e:
        logger.error(f"Erreur chargement {model_name}: {e}")
        raise


def get_available_models() -> List[str]:
    """Retourne la liste des modèles YOLOv8 disponibles."""
    return ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]


if __name__ == "__main__":
    # Test des fonctionnalités
    manager = ModelManager()

    print("✅ Module model_utils initialisé avec succès!")
    print(f"🔧 Device détecté: {manager.device}")
    print(f"📋 Configuration: {manager.config['model']['name']}")
    print(f"🎯 Classes: {list(manager.config['model']['class_names'].values())}")
