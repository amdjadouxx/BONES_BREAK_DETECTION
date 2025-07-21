"""
Script de préparation et téléchargement des données pour la détection de fractures pédiatriques.
"""

import argparse
import os
import sys
from pathlib import Path
import requests
import zipfile
import tarfile
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import yaml
from loguru import logger

# Ajouter le répertoire parent au path
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_utils import DataProcessor


class DatasetDownloader:
    """Gestionnaire de téléchargement et préparation des datasets."""

    def __init__(self, config_path: str = "config/data_config.yaml"):
        """
        Initialise le téléchargeur.

        Args:
            config_path: Chemin vers la configuration des données
        """
        self.config = self.load_config(config_path)
        self.data_processor = DataProcessor(config_path)
        self.base_path = Path(self.config["dataset"]["paths"]["raw"])
        self.base_path.mkdir(parents=True, exist_ok=True)

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Charge la configuration des données."""
        try:
            with open(config_path, "r", encoding="utf-8") as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"Configuration non trouvée: {config_path}")
            return self.get_default_config()

    def get_default_config(self) -> Dict[str, Any]:
        """Configuration par défaut."""
        return {
            "dataset": {
                "name": "pediatric_fractures",
                "paths": {
                    "raw": "data/raw",
                    "processed": "data/processed",
                    "annotations": "data/annotations",
                },
                "sources": {
                    "rsna": {
                        "description": "RSNA Pediatric Bone Age Dataset",
                        "size": "~12GB",
                    },
                    "mura": {"description": "MURA Dataset", "size": "~40GB"},
                },
            }
        }

    def download_file(self, url: str, filepath: str, chunk_size: int = 8192) -> bool:
        """
        Télécharge un fichier avec barre de progression.

        Args:
            url: URL du fichier
            filepath: Chemin de destination
            chunk_size: Taille des chunks de téléchargement

        Returns:
            True si succès, False sinon
        """
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            with open(filepath, "wb") as file, tqdm(
                desc=f"📥 {Path(filepath).name}",
                total=total_size,
                unit="B",
                unit_scale=True,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        file.write(chunk)
                        pbar.update(len(chunk))

            logger.info(f"✅ Téléchargé: {filepath}")
            return True

        except Exception as e:
            logger.error(f"❌ Erreur téléchargement {url}: {e}")
            return False

    def extract_archive(self, archive_path: str, extract_to: str) -> bool:
        """
        Extrait une archive (zip, tar, tar.gz).

        Args:
            archive_path: Chemin vers l'archive
            extract_to: Répertoire d'extraction

        Returns:
            True si succès, False sinon
        """
        try:
            archive_path = Path(archive_path)
            extract_to = Path(extract_to)
            extract_to.mkdir(parents=True, exist_ok=True)

            logger.info(f"📦 Extraction: {archive_path.name}")

            if archive_path.suffix.lower() == ".zip":
                with zipfile.ZipFile(archive_path, "r") as zip_ref:
                    zip_ref.extractall(extract_to)

            elif archive_path.suffix.lower() in [".tar", ".gz"]:
                with tarfile.open(archive_path, "r:*") as tar_ref:
                    tar_ref.extractall(extract_to)

            else:
                raise ValueError(
                    f"Format d'archive non supporté: {archive_path.suffix}"
                )

            logger.info(f"✅ Extrait dans: {extract_to}")
            return True

        except Exception as e:
            logger.error(f"❌ Erreur extraction {archive_path}: {e}")
            return False

    def download_sample_dataset(self) -> bool:
        """
        Télécharge un dataset d'exemple pour la démonstration.

        Returns:
            True si succès, False sinon
        """
        logger.info("📥 Téléchargement du dataset d'exemple...")

        # URLs d'exemples (remplacer par de vraies URLs)
        sample_urls = [
            "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip",
            # Ajouter d'autres sources réelles ici
        ]

        sample_dir = self.base_path / "sample_data"
        sample_dir.mkdir(exist_ok=True)

        success = True
        for i, url in enumerate(sample_urls):
            filename = f"sample_dataset_{i}.zip"
            filepath = sample_dir / filename

            if not filepath.exists():
                if not self.download_file(url, str(filepath)):
                    success = False
                    continue

                # Extraire l'archive
                extract_dir = sample_dir / f"extracted_{i}"
                if not self.extract_archive(str(filepath), str(extract_dir)):
                    success = False

        return success

    def create_synthetic_dataset(self, num_images: int = 100) -> bool:
        """
        Crée un dataset synthétique pour les tests.

        Args:
            num_images: Nombre d'images à créer

        Returns:
            True si succès, False sinon
        """
        logger.info(f"🎨 Création d'un dataset synthétique ({num_images} images)...")

        try:
            import cv2

            synthetic_dir = self.base_path / "synthetic_data"
            synthetic_dir.mkdir(exist_ok=True)

            images_dir = synthetic_dir / "images"
            labels_dir = synthetic_dir / "labels"
            images_dir.mkdir(exist_ok=True)
            labels_dir.mkdir(exist_ok=True)

            # Créer des images synthétiques
            for i in tqdm(range(num_images), desc="🎨 Génération d'images"):
                # Image synthétique (radiographie simulée)
                img = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)

                # Ajouter du bruit pour simuler une radiographie
                noise = np.random.normal(0, 25, img.shape)
                img = np.clip(img + noise, 0, 255).astype(np.uint8)

                # Convertir en niveaux de gris (comme une vraie radiographie)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

                # Sauvegarder l'image
                img_path = images_dir / f"synthetic_{i:04d}.jpg"
                cv2.imwrite(str(img_path), img)

                # Créer une annotation YOLO synthétique (optionnel)
                if np.random.random() < 0.3:  # 30% des images ont une "fracture"
                    # Générer des coordonnées aléatoires pour la bounding box
                    x_center = np.random.uniform(0.2, 0.8)
                    y_center = np.random.uniform(0.2, 0.8)
                    width = np.random.uniform(0.1, 0.3)
                    height = np.random.uniform(0.1, 0.3)

                    # Annotation YOLO format
                    label_path = labels_dir / f"synthetic_{i:04d}.txt"
                    with open(label_path, "w") as f:
                        f.write(
                            f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                        )

            logger.info(f"✅ Dataset synthétique créé: {num_images} images")

            # Créer un fichier de métadonnées
            metadata = {
                "dataset_type": "synthetic",
                "num_images": num_images,
                "image_size": [640, 640],
                "classes": ["fracture"],
                "format": "YOLO",
                "created_by": "DatasetDownloader",
            }

            metadata_path = synthetic_dir / "metadata.yaml"
            with open(metadata_path, "w", encoding="utf-8") as f:
                yaml.dump(metadata, f, default_flow_style=False)

            return True

        except Exception as e:
            logger.error(f"❌ Erreur création dataset synthétique: {e}")
            return False

    def prepare_rsna_dataset(self, data_dir: str) -> bool:
        """
        Prépare le dataset RSNA Pediatric Bone Age.

        Args:
            data_dir: Répertoire contenant les données RSNA

        Returns:
            True si succès, False sinon
        """
        logger.info("🏥 Préparation du dataset RSNA...")

        try:
            data_path = Path(data_dir)
            if not data_path.exists():
                logger.error(f"Répertoire RSNA non trouvé: {data_dir}")
                return False

            # Chercher les fichiers CSV de métadonnées
            csv_files = list(data_path.glob("*.csv"))
            if not csv_files:
                logger.error("Aucun fichier CSV de métadonnées trouvé")
                return False

            # Charger les métadonnées
            metadata_file = csv_files[0]
            df = pd.read_csv(metadata_file)
            logger.info(f"📊 Métadonnées chargées: {len(df)} échantillons")

            # Créer la structure de données
            processed_dir = Path(self.config["dataset"]["paths"]["processed"])

            # Diviser en train/val/test
            split_config = self.config.get(
                "data_split", {"train": 0.7, "val": 0.2, "test": 0.1}
            )

            train_size = int(len(df) * split_config["train"])
            val_size = int(len(df) * split_config["val"])

            # Mélanger les données
            df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

            # Diviser
            train_df = df_shuffled[:train_size]
            val_df = df_shuffled[train_size : train_size + val_size]
            test_df = df_shuffled[train_size + val_size :]

            # Sauvegarder les splits
            for split_name, split_df in [
                ("train", train_df),
                ("val", val_df),
                ("test", test_df),
            ]:
                split_dir = processed_dir / split_name
                split_dir.mkdir(parents=True, exist_ok=True)

                split_csv = split_dir / "metadata.csv"
                split_df.to_csv(split_csv, index=False)
                logger.info(f"📁 {split_name}: {len(split_df)} échantillons")

            logger.info("✅ Dataset RSNA préparé avec succès")
            return True

        except Exception as e:
            logger.error(f"❌ Erreur préparation RSNA: {e}")
            return False

    def create_yolo_dataset_config(
        self, dataset_dir: str, output_path: str = "data/dataset.yaml"
    ) -> bool:
        """
        Crée un fichier de configuration YOLO pour le dataset.

        Args:
            dataset_dir: Répertoire du dataset
            output_path: Chemin du fichier de configuration

        Returns:
            True si succès, False sinon
        """
        try:
            dataset_path = Path(dataset_dir).resolve()

            config = {
                "path": str(dataset_path),
                "train": "train/images",
                "val": "val/images",
                "test": "test/images",
                "nc": 1,  # Nombre de classes
                "names": ["fracture"],  # Noms des classes
            }

            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False)

            logger.info(f"✅ Configuration YOLO créée: {output_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Erreur création config YOLO: {e}")
            return False

    def validate_dataset_structure(self, dataset_dir: str) -> Dict[str, Any]:
        """
        Valide la structure d'un dataset.

        Args:
            dataset_dir: Répertoire du dataset

        Returns:
            Rapport de validation
        """
        logger.info(f"🔍 Validation de la structure: {dataset_dir}")

        dataset_path = Path(dataset_dir)

        report = {"valid": True, "issues": [], "statistics": {}, "structure": {}}

        try:
            # Vérifier la structure de base
            required_dirs = ["train", "val", "test"]
            for split in required_dirs:
                split_path = dataset_path / split
                if not split_path.exists():
                    report["issues"].append(f"Répertoire manquant: {split}")
                    report["valid"] = False
                else:
                    images_path = split_path / "images"
                    labels_path = split_path / "labels"

                    if images_path.exists():
                        image_count = len(list(images_path.glob("*.*")))
                        report["structure"][f"{split}_images"] = image_count
                    else:
                        report["issues"].append(
                            f"Répertoire images manquant: {split}/images"
                        )

                    if labels_path.exists():
                        label_count = len(list(labels_path.glob("*.txt")))
                        report["structure"][f"{split}_labels"] = label_count
                    else:
                        report["issues"].append(
                            f"Répertoire labels manquant: {split}/labels"
                        )

            # Statistiques globales
            total_images = sum(
                v for k, v in report["structure"].items() if "images" in k
            )
            total_labels = sum(
                v for k, v in report["structure"].items() if "labels" in k
            )

            report["statistics"] = {
                "total_images": total_images,
                "total_labels": total_labels,
                "annotation_rate": (
                    total_labels / total_images if total_images > 0 else 0
                ),
            }

            logger.info(
                f"📊 Validation terminée: {len(report['issues'])} problème(s) trouvé(s)"
            )

        except Exception as e:
            report["valid"] = False
            report["issues"].append(f"Erreur validation: {str(e)}")

        return report

    def show_dataset_info(self):
        """Affiche les informations sur les datasets disponibles."""
        logger.info("📚 Datasets disponibles:")

        sources = self.config.get("dataset", {}).get("sources", {})
        for name, info in sources.items():
            logger.info(f"  • {name.upper()}: {info.get('description', 'N/A')}")
            logger.info(f"    Taille: {info.get('size', 'Inconnue')}")
            if "url" in info:
                logger.info(f"    URL: {info['url']}")
            logger.info("")


def parse_arguments():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="🏥 Préparation des données pour la détection de fractures pédiatriques",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  # Télécharger dataset d'exemple
  python scripts/prepare_data.py --download --sample
  
  # Créer un dataset synthétique
  python scripts/prepare_data.py --synthetic --num-images 500
  
  # Préparer dataset RSNA existant
  python scripts/prepare_data.py --prepare-rsna /path/to/rsna/data
  
  # Valider un dataset
  python scripts/prepare_data.py --validate /path/to/dataset
        """,
    )

    parser.add_argument(
        "--download", action="store_true", help="Télécharger les données"
    )
    parser.add_argument(
        "--sample", action="store_true", help="Télécharger le dataset d'exemple"
    )
    parser.add_argument(
        "--synthetic", action="store_true", help="Créer un dataset synthétique"
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=100,
        help="Nombre d'images synthétiques (défaut: 100)",
    )
    parser.add_argument(
        "--prepare-rsna",
        type=str,
        help="Préparer dataset RSNA depuis le répertoire donné",
    )
    parser.add_argument(
        "--validate", type=str, help="Valider la structure d'un dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/data_config.yaml",
        help="Fichier de configuration",
    )
    parser.add_argument(
        "--info", action="store_true", help="Afficher les informations des datasets"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Mode verbeux")

    return parser.parse_args()


def setup_logging(verbose: bool = False):
    """Configure le logging."""
    level = "DEBUG" if verbose else "INFO"

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level=level,
    )


def main():
    """Fonction principale."""
    args = parse_arguments()
    setup_logging(args.verbose)

    logger.info("🏥 Préparation des données pour détection de fractures pédiatriques")

    try:
        # Initialiser le downloader
        downloader = DatasetDownloader(args.config)

        # Afficher les informations si demandé
        if args.info:
            downloader.show_dataset_info()
            return

        # Télécharger données d'exemple
        if args.download and args.sample:
            logger.info("📥 Téléchargement du dataset d'exemple...")
            if downloader.download_sample_dataset():
                logger.info("✅ Dataset d'exemple téléchargé avec succès")
            else:
                logger.error("❌ Échec téléchargement dataset d'exemple")

        # Créer dataset synthétique
        if args.synthetic:
            logger.info(f"🎨 Création de {args.num_images} images synthétiques...")
            if downloader.create_synthetic_dataset(args.num_images):
                logger.info("✅ Dataset synthétique créé avec succès")

                # Créer la configuration YOLO
                synthetic_dir = downloader.base_path / "synthetic_data"
                downloader.create_yolo_dataset_config(str(synthetic_dir))
            else:
                logger.error("❌ Échec création dataset synthétique")

        # Préparer dataset RSNA
        if args.prepare_rsna:
            logger.info(f"🏥 Préparation dataset RSNA: {args.prepare_rsna}")
            if downloader.prepare_rsna_dataset(args.prepare_rsna):
                logger.info("✅ Dataset RSNA préparé avec succès")
            else:
                logger.error("❌ Échec préparation dataset RSNA")

        # Valider dataset
        if args.validate:
            logger.info(f"🔍 Validation du dataset: {args.validate}")
            report = downloader.validate_dataset_structure(args.validate)

            if report["valid"]:
                logger.info("✅ Structure du dataset valide")
            else:
                logger.warning("⚠️  Problèmes détectés dans la structure:")
                for issue in report["issues"]:
                    logger.warning(f"  • {issue}")

            # Afficher les statistiques
            stats = report["statistics"]
            logger.info(f"📊 Statistiques:")
            logger.info(f"  • Images totales: {stats.get('total_images', 0)}")
            logger.info(f"  • Labels totaux: {stats.get('total_labels', 0)}")
            logger.info(f"  • Taux d'annotation: {stats.get('annotation_rate', 0):.2%}")

        logger.info("🎉 Préparation des données terminée!")

    except KeyboardInterrupt:
        logger.warning("⏹️  Opération interrompue par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
