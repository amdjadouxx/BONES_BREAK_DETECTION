"""
Script principal pour la détection de fractures pédiatriques avec YOLOv8.
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from loguru import logger

# Ajouter le répertoire parent au path pour importer les modules
sys.path.append(str(Path(__file__).parent.parent))

from utils.model_utils import ModelManager, PredictionPostProcessor
from utils.visualization import MedicalImageVisualizer
from utils.data_utils import DataProcessor


class PediatricFractureDetector:
    """Détecteur de fractures pédiatriques utilisant YOLOv8."""

    def __init__(
        self, model_path: str = None, config_path: str = "config/model_config.yaml"
    ):
        """
        Initialise le détecteur.

        Args:
            model_path: Chemin vers le modèle personnalisé (si None, utilise le pré-entraîné)
            config_path: Chemin vers la configuration du modèle
        """
        self.model_manager = ModelManager(config_path)
        self.post_processor = PredictionPostProcessor(
            self.model_manager.config["model"]["class_names"]
        )
        self.visualizer = MedicalImageVisualizer()

        # Charger le modèle
        try:
            self.model = self.model_manager.load_model(model_path)
            logger.info("✅ Détecteur initialisé avec succès")
        except Exception as e:
            logger.error(f"❌ Erreur initialisation: {e}")
            raise

    def predict_single_image(
        self,
        image_path: str,
        confidence: float = None,
        save_results: bool = False,
        output_dir: str = "results",
    ) -> Dict[str, Any]:
        """
        Effectue une prédiction sur une image unique.

        Args:
            image_path: Chemin vers l'image
            confidence: Seuil de confiance (utilise config si None)
            save_results: Sauvegarder les résultats
            output_dir: Répertoire de sortie

        Returns:
            Dictionnaire avec les résultats de détection
        """
        try:
            # Vérifier que l'image existe
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image non trouvée: {image_path}")

            # Prédiction
            results = self.model_manager.predict(
                source=image_path,
                conf=confidence,
                save=False,  # On gère la sauvegarde manuellement
            )

            # Post-traitement
            processed_results = self.post_processor.process_results(results)

            if not processed_results:
                logger.warning(f"Aucun résultat pour {image_path}")
                return {
                    "image_path": image_path,
                    "detections": [],
                    "detection_count": 0,
                    "status": "no_detections",
                }

            result = processed_results[0]  # Une seule image

            # Sauvegarder si demandé
            if save_results:
                self.save_single_result(image_path, result, output_dir)

            # Log du résultat
            logger.info(
                f"🎯 {result['detection_count']} fracture(s) détectée(s) dans {Path(image_path).name}"
            )

            return {**result, "status": "success"}

        except Exception as e:
            logger.error(f"❌ Erreur prédiction {image_path}: {e}")
            return {
                "image_path": image_path,
                "detections": [],
                "detection_count": 0,
                "status": "error",
                "error": str(e),
            }

    def predict_batch(
        self,
        image_paths: List[str],
        confidence: float = None,
        save_results: bool = False,
        output_dir: str = "results",
    ) -> List[Dict[str, Any]]:
        """
        Effectue des prédictions sur plusieurs images.

        Args:
            image_paths: Liste des chemins d'images
            confidence: Seuil de confiance
            save_results: Sauvegarder les résultats
            output_dir: Répertoire de sortie

        Returns:
            Liste des résultats pour chaque image
        """
        results = []

        logger.info(f"🔍 Traitement de {len(image_paths)} images...")

        for i, image_path in enumerate(image_paths, 1):
            logger.info(
                f"📸 [{i}/{len(image_paths)}] Traitement: {Path(image_path).name}"
            )

            result = self.predict_single_image(
                image_path, confidence, save_results, output_dir
            )
            results.append(result)

        # Statistiques globales
        total_detections = sum(r["detection_count"] for r in results)
        images_with_detections = sum(1 for r in results if r["detection_count"] > 0)

        logger.info(f"✅ Traitement terminé:")
        logger.info(f"   • {total_detections} fractures détectées au total")
        logger.info(
            f"   • {images_with_detections}/{len(image_paths)} images avec détections"
        )

        return results

    def predict_directory(
        self,
        directory_path: str,
        extensions: List[str] = None,
        confidence: float = None,
        save_results: bool = False,
        output_dir: str = "results",
    ) -> List[Dict[str, Any]]:
        """
        Effectue des prédictions sur toutes les images d'un répertoire.

        Args:
            directory_path: Chemin vers le répertoire
            extensions: Extensions d'images à traiter
            confidence: Seuil de confiance
            save_results: Sauvegarder les résultats
            output_dir: Répertoire de sortie

        Returns:
            Liste des résultats
        """
        if extensions is None:
            extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

        # Trouver toutes les images
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Répertoire non trouvé: {directory_path}")

        image_paths = []
        for ext in extensions:
            image_paths.extend(directory.glob(f"*{ext}"))
            image_paths.extend(directory.glob(f"*{ext.upper()}"))

        image_paths = [str(p) for p in image_paths]

        if not image_paths:
            logger.warning(f"Aucune image trouvée dans {directory_path}")
            return []

        logger.info(f"📁 Répertoire: {directory_path} ({len(image_paths)} images)")

        return self.predict_batch(image_paths, confidence, save_results, output_dir)

    def save_single_result(
        self, image_path: str, result: Dict[str, Any], output_dir: str
    ):
        """
        Sauvegarde les résultats d'une prédiction unique.

        Args:
            image_path: Chemin de l'image source
            result: Résultats de la prédiction
            output_dir: Répertoire de sortie
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        image_name = Path(image_path).stem

        # 1. Image avec bounding boxes
        if result["detections"]:
            annotated_path = output_path / f"{image_name}_detected.jpg"
            self.visualizer.draw_bboxes_on_image(
                image_path, result["detections"], str(annotated_path)
            )

        # 2. Rapport texte détaillé
        report_path = output_path / f"{image_name}_report.txt"
        self.save_detection_report(image_path, result, str(report_path))

        # 3. Résultats JSON
        import json

        json_path = output_path / f"{image_name}_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            # Préparer les données pour JSON (convertir numpy arrays)
            json_result = {
                "image_path": image_path,
                "detection_count": result["detection_count"],
                "detections": [],
            }

            for detection in result["detections"]:
                json_detection = {
                    "class_name": detection["class_name"],
                    "confidence": float(detection["confidence"]),
                    "bbox": {k: float(v) for k, v in detection["bbox"].items()},
                }
                json_result["detections"].append(json_detection)

            json.dump(json_result, f, indent=2, ensure_ascii=False)

    def save_detection_report(
        self, image_path: str, result: Dict[str, Any], report_path: str
    ):
        """Sauvegarde un rapport détaillé de la détection."""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("🦴 RAPPORT DE DÉTECTION DE FRACTURES PÉDIATRIQUES\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"📁 Image source: {image_path}\n")
            f.write(f"📊 Nombre de détections: {result['detection_count']}\n")
            f.write(f"📅 Analysé avec YOLOv8\n\n")

            if result["detections"]:
                f.write("🎯 DÉTECTIONS:\n")
                f.write("-" * 20 + "\n")

                for i, detection in enumerate(result["detections"], 1):
                    f.write(f"\n📍 Détection #{i}:\n")
                    f.write(f"   • Type: {detection['class_name']}\n")
                    f.write(
                        f"   • Confiance: {detection['confidence']:.3f} ({detection['confidence']*100:.1f}%)\n"
                    )
                    f.write(
                        f"   • Position: ({detection['bbox']['x1']:.0f}, {detection['bbox']['y1']:.0f}) "
                        f"-> ({detection['bbox']['x2']:.0f}, {detection['bbox']['y2']:.0f})\n"
                    )
                    f.write(
                        f"   • Taille: {detection['bbox']['width']:.0f} x {detection['bbox']['height']:.0f} pixels\n"
                    )
                    f.write(
                        f"   • Centre: ({detection['bbox']['center_x']:.0f}, {detection['bbox']['center_y']:.0f})\n"
                    )

                # Analyse des résultats
                f.write(f"\n📈 ANALYSE:\n")
                f.write("-" * 15 + "\n")
                avg_confidence = np.mean(
                    [d["confidence"] for d in result["detections"]]
                )
                f.write(f"• Confiance moyenne: {avg_confidence:.3f}\n")

                high_confidence = sum(
                    1 for d in result["detections"] if d["confidence"] > 0.7
                )
                f.write(f"• Détections haute confiance (>70%): {high_confidence}\n")

                total_area = sum(
                    d["bbox"]["width"] * d["bbox"]["height"]
                    for d in result["detections"]
                )
                f.write(f"• Aire totale des fractures: {total_area:.0f} pixels²\n")
            else:
                f.write("✅ AUCUNE FRACTURE DÉTECTÉE\n")
                f.write(
                    "L'analyse n'a révélé aucune fracture visible avec le seuil de confiance configuré.\n"
                )

            f.write(f"\n" + "=" * 60 + "\n")
            f.write("⚠️  AVERTISSEMENT MÉDICAL:\n")
            f.write(
                "Ce rapport est généré par un système automatisé d'aide au diagnostic.\n"
            )
            f.write(
                "Il ne remplace pas l'expertise d'un radiologue ou médecin qualifié.\n"
            )
            f.write(
                "Toujours consulter un professionnel de santé pour un diagnostic définitif.\n"
            )

    def get_model_info(self) -> Dict[str, Any]:
        """Retourne les informations du modèle chargé."""
        return self.model_manager.get_model_info()


def parse_arguments():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="🦴 Détection de fractures pédiatriques avec YOLOv8",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  # Image unique
  python inference/predict.py --source image.jpg --output results/
  
  # Répertoire d'images  
  python inference/predict.py --source images/ --output results/ --save
  
  # Avec seuil personnalisé
  python inference/predict.py --source image.jpg --confidence 0.7 --save
        """,
    )

    parser.add_argument(
        "--source", type=str, required=True, help="Source: image, répertoire"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Chemin vers modèle personnalisé (optionnel)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/model_config.yaml",
        help="Fichier de configuration",
    )
    parser.add_argument(
        "--confidence", type=float, default=None, help="Seuil de confiance (0-1)"
    )
    parser.add_argument(
        "--output", type=str, default="results", help="Répertoire de sortie"
    )
    parser.add_argument("--save", action="store_true", help="Sauvegarder les résultats")
    parser.add_argument("--verbose", "-v", action="store_true", help="Mode verbeux")

    return parser.parse_args()


def setup_logging(verbose: bool = False):
    """Configure le logging."""
    level = "DEBUG" if verbose else "INFO"

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level=level,
    )


def main():
    """Fonction principale."""
    args = parse_arguments()
    setup_logging(args.verbose)

    logger.info("🦴 Démarrage du détecteur de fractures pédiatriques")

    try:
        # Initialiser le détecteur
        detector = PediatricFractureDetector(
            model_path=args.model, config_path=args.config
        )

        # Afficher les infos du modèle
        model_info = detector.get_model_info()
        logger.info(f"📋 Modèle: {model_info.get('model_name', 'N/A')}")
        logger.info(f"🎯 Classes: {list(model_info.get('class_names', {}).values())}")
        logger.info(f"💻 Device: {model_info.get('device', 'N/A')}")

        # Déterminer le type de source
        source_path = Path(args.source)

        if source_path.is_file():
            # Image unique
            logger.info(f"📸 Analyse d'une image: {args.source}")
            result = detector.predict_single_image(
                args.source,
                confidence=args.confidence,
                save_results=args.save,
                output_dir=args.output,
            )

            if result["status"] == "success":
                if result["detection_count"] > 0:
                    logger.info(
                        f"✅ {result['detection_count']} fracture(s) détectée(s)"
                    )
                else:
                    logger.info("✅ Aucune fracture détectée")
            else:
                logger.error(f"❌ Erreur: {result.get('error', 'Inconnue')}")

        elif source_path.is_dir():
            # Répertoire d'images
            logger.info(f"📁 Analyse du répertoire: {args.source}")
            results = detector.predict_directory(
                args.source,
                confidence=args.confidence,
                save_results=args.save,
                output_dir=args.output,
            )

            # Statistiques finales
            total_images = len(results)
            successful = sum(1 for r in results if r["status"] == "success")
            total_detections = sum(
                r["detection_count"] for r in results if r["status"] == "success"
            )

            logger.info(f"📊 Résumé final:")
            logger.info(f"   • Images traitées: {successful}/{total_images}")
            logger.info(f"   • Fractures détectées: {total_detections}")

        else:
            raise FileNotFoundError(f"Source non trouvée: {args.source}")

        if args.save:
            logger.info(f"💾 Résultats sauvés dans: {args.output}")

        logger.info("🎉 Analyse terminée avec succès!")

    except KeyboardInterrupt:
        logger.warning("⏹️  Analyse interrompue par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
