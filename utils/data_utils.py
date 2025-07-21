"""
Utilitaires pour la gestion et le prétraitement des données médicales.
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import pydicom
from PIL import Image
import yaml
from loguru import logger


class DataProcessor:
    """Classe pour le prétraitement des données médicales."""

    def __init__(self, config_path: str = "config/data_config.yaml"):
        """
        Initialise le processeur de données.

        Args:
            config_path: Chemin vers le fichier de configuration
        """
        self.config = self.load_config(config_path)
        self.setup_directories()

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Charge la configuration depuis un fichier YAML."""
        try:
            with open(config_path, "r", encoding="utf-8") as file:
                config = yaml.safe_load(file)
            logger.info(f"Configuration chargée depuis {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Fichier de config non trouvé: {config_path}")
            return self.get_default_config()

    def get_default_config(self) -> Dict[str, Any]:
        """Configuration par défaut."""
        return {
            "preprocessing": {
                "target_size": [640, 640],
                "output_format": "jpg",
                "normalization": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                },
            },
            "dataset": {
                "paths": {
                    "raw": "data/raw",
                    "processed": "data/processed",
                    "annotations": "data/annotations",
                }
            },
        }

    def setup_directories(self):
        """Crée les répertoires nécessaires."""
        paths = self.config.get("dataset", {}).get("paths", {})
        for path in paths.values():
            os.makedirs(path, exist_ok=True)
            logger.info(f"Répertoire créé/vérifié: {path}")

    def convert_dicom_to_jpg(self, dicom_path: str, output_path: str) -> bool:
        """
        Convertit un fichier DICOM en JPG.

        Args:
            dicom_path: Chemin vers le fichier DICOM
            output_path: Chemin de sortie JPG

        Returns:
            True si succès, False sinon
        """
        try:
            # Lire le fichier DICOM
            dicom = pydicom.dcmread(dicom_path)

            # Extraire les pixels
            pixel_array = dicom.pixel_array

            # Normalisation et windowing pour radiographies
            if hasattr(dicom, "WindowCenter") and hasattr(dicom, "WindowWidth"):
                window_center = dicom.WindowCenter
                window_width = dicom.WindowWidth
            else:
                # Valeurs par défaut pour radiographies osseuses
                window_center = self.config["preprocessing"]["dicom_conversion"][
                    "window_center"
                ]
                window_width = self.config["preprocessing"]["dicom_conversion"][
                    "window_width"
                ]

            # Appliquer le windowing
            img_min = window_center - window_width // 2
            img_max = window_center + window_width // 2

            pixel_array = np.clip(pixel_array, img_min, img_max)
            pixel_array = ((pixel_array - img_min) / (img_max - img_min) * 255).astype(
                np.uint8
            )

            # Convertir en PIL Image et sauvegarder
            image = Image.fromarray(pixel_array)
            if image.mode != "RGB":
                image = image.convert("RGB")

            image.save(output_path, "JPEG", quality=95)
            logger.debug(f"DICOM converti: {dicom_path} -> {output_path}")
            return True

        except Exception as e:
            logger.error(f"Erreur conversion DICOM {dicom_path}: {e}")
            return False

    def resize_image(
        self, image_path: str, output_path: str, target_size: Tuple[int, int] = None
    ) -> bool:
        """
        Redimensionne une image.

        Args:
            image_path: Chemin vers l'image source
            output_path: Chemin de sortie
            target_size: Taille cible (width, height)

        Returns:
            True si succès, False sinon
        """
        if target_size is None:
            target_size = tuple(self.config["preprocessing"]["target_size"])

        try:
            # Charger l'image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Impossible de charger l'image: {image_path}")

            # Redimensionner en conservant le ratio
            h, w = image.shape[:2]
            target_w, target_h = target_size

            # Calculer le ratio pour conserver les proportions
            ratio = min(target_w / w, target_h / h)
            new_w = int(w * ratio)
            new_h = int(h * ratio)

            # Redimensionner
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Créer une image finale avec padding si nécessaire
            final_image = np.zeros((target_h, target_w, 3), dtype=np.uint8)

            # Centrer l'image redimensionnée
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            final_image[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = (
                resized
            )

            # Sauvegarder
            cv2.imwrite(output_path, final_image)
            logger.debug(f"Image redimensionnée: {image_path} -> {output_path}")
            return True

        except Exception as e:
            logger.error(f"Erreur redimensionnement {image_path}: {e}")
            return False

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalise une image selon les paramètres de configuration.

        Args:
            image: Image à normaliser (format BGR ou RGB)

        Returns:
            Image normalisée
        """
        mean = np.array(self.config["preprocessing"]["normalization"]["mean"])
        std = np.array(self.config["preprocessing"]["normalization"]["std"])

        # Convertir en float et normaliser [0, 1]
        image = image.astype(np.float32) / 255.0

        # Appliquer la normalisation ImageNet
        normalized = (image - mean) / std

        return normalized

    def create_metadata_df(self, data_dir: str) -> pd.DataFrame:
        """
        Crée un DataFrame avec les métadonnées des images.

        Args:
            data_dir: Répertoire contenant les images

        Returns:
            DataFrame avec les métadonnées
        """
        metadata = []
        supported_formats = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

        for image_path in Path(data_dir).rglob("*"):
            if image_path.suffix.lower() in supported_formats:
                try:
                    # Informations de base
                    info = {
                        "filename": image_path.name,
                        "path": str(image_path),
                        "size_bytes": image_path.stat().st_size,
                    }

                    # Dimensions de l'image
                    image = cv2.imread(str(image_path))
                    if image is not None:
                        h, w, c = image.shape
                        info.update(
                            {
                                "width": w,
                                "height": h,
                                "channels": c,
                                "aspect_ratio": w / h,
                            }
                        )

                    metadata.append(info)

                except Exception as e:
                    logger.warning(f"Erreur métadonnées pour {image_path}: {e}")

        df = pd.DataFrame(metadata)
        logger.info(f"Métadonnées créées pour {len(df)} images")
        return df

    def validate_dataset(
        self, data_dir: str, annotation_dir: str = None
    ) -> Dict[str, Any]:
        """
        Valide un dataset et retourne un rapport.

        Args:
            data_dir: Répertoire des images
            annotation_dir: Répertoire des annotations (optionnel)

        Returns:
            Dictionnaire contenant le rapport de validation
        """
        report = {
            "total_images": 0,
            "valid_images": 0,
            "invalid_images": [],
            "annotations_found": 0,
            "missing_annotations": [],
            "issues": [],
        }

        # Vérifier les images
        supported_formats = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

        for image_path in Path(data_dir).rglob("*"):
            if image_path.suffix.lower() in supported_formats:
                report["total_images"] += 1

                try:
                    # Vérifier que l'image peut être chargée
                    image = cv2.imread(str(image_path))
                    if image is None:
                        report["invalid_images"].append(str(image_path))
                        continue

                    # Vérifier les dimensions minimales
                    h, w = image.shape[:2]
                    min_size = (
                        self.config.get("validation", {})
                        .get("quality_thresholds", {})
                        .get("min_image_size", [256, 256])
                    )

                    if w < min_size[0] or h < min_size[1]:
                        report["issues"].append(
                            f"Image trop petite: {image_path} ({w}x{h})"
                        )

                    report["valid_images"] += 1

                    # Vérifier l'annotation correspondante si annotation_dir fourni
                    if annotation_dir:
                        annotation_path = (
                            Path(annotation_dir) / f"{image_path.stem}.txt"
                        )
                        if annotation_path.exists():
                            report["annotations_found"] += 1
                        else:
                            report["missing_annotations"].append(str(image_path))

                except Exception as e:
                    report["invalid_images"].append(str(image_path))
                    logger.error(f"Erreur validation image {image_path}: {e}")

        logger.info(
            f"Validation terminée: {report['valid_images']}/{report['total_images']} images valides"
        )
        return report


def load_dataset_config(config_path: str = "config/data_config.yaml") -> Dict[str, Any]:
    """Charge la configuration du dataset."""
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_dataset_statistics(data_dir: str) -> Dict[str, Any]:
    """
    Calcule les statistiques du dataset.

    Args:
        data_dir: Répertoire du dataset

    Returns:
        Dictionnaire avec les statistiques
    """
    stats = {
        "total_images": 0,
        "total_size_mb": 0,
        "image_formats": {},
        "dimensions": [],
        "mean_rgb": [0, 0, 0],
        "std_rgb": [0, 0, 0],
    }

    supported_formats = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    pixel_values = []

    for image_path in Path(data_dir).rglob("*"):
        if image_path.suffix.lower() in supported_formats:
            try:
                stats["total_images"] += 1
                stats["total_size_mb"] += image_path.stat().st_size / (1024 * 1024)

                # Format
                fmt = image_path.suffix.lower()
                stats["image_formats"][fmt] = stats["image_formats"].get(fmt, 0) + 1

                # Dimensions et pixels
                image = cv2.imread(str(image_path))
                if image is not None:
                    h, w = image.shape[:2]
                    stats["dimensions"].append((w, h))

                    # Échantillonner quelques pixels pour les statistiques
                    if len(pixel_values) < 1000:  # Limiter pour la performance
                        sample = cv2.resize(image, (64, 64))
                        pixel_values.extend(sample.reshape(-1, 3))

            except Exception as e:
                logger.warning(f"Erreur statistiques pour {image_path}: {e}")

    # Calculer mean et std
    if pixel_values:
        pixel_array = np.array(pixel_values)
        stats["mean_rgb"] = np.mean(pixel_array, axis=0).tolist()
        stats["std_rgb"] = np.std(pixel_array, axis=0).tolist()

    logger.info(f"Statistiques calculées pour {stats['total_images']} images")
    return stats


if __name__ == "__main__":
    # Test des fonctionnalités
    processor = DataProcessor()

    # Exemple d'utilisation
    print("✅ Module data_utils initialisé avec succès!")
    print(f"📁 Configuration chargée")
    print(f"📊 Prêt pour le prétraitement des données médicales")
