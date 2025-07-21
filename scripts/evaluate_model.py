"""
Script d'évaluation des performances du modèle YOLOv8 pour la détection de fractures.
"""

import argparse
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

# Ajouter le répertoire parent au path
sys.path.append(str(Path(__file__).parent.parent))

from utils.model_utils import ModelManager
from utils.metrics import DetectionMetrics, ClassificationMetrics
from utils.visualization import MedicalImageVisualizer, create_metrics_dashboard


class ModelEvaluator:
    """Évaluateur de performance pour les modèles de détection de fractures."""

    def __init__(self, model_path: str, config_path: str = "config/model_config.yaml"):
        """
        Initialise l'évaluateur.

        Args:
            model_path: Chemin vers le modèle entraîné
            config_path: Chemin vers la configuration
        """
        self.model_manager = ModelManager(config_path)
        self.detection_metrics = DetectionMetrics()
        self.classification_metrics = ClassificationMetrics()
        self.visualizer = MedicalImageVisualizer()

        # Charger le modèle
        try:
            self.model = self.model_manager.load_model(model_path)
            logger.info(f"✅ Modèle chargé: {model_path}")
        except Exception as e:
            logger.error(f"❌ Erreur chargement modèle: {e}")
            raise

    def evaluate_on_test_set(
        self, test_data_path: str, confidence_threshold: float = 0.25
    ) -> Dict[str, Any]:
        """
        Évalue le modèle sur le jeu de test.

        Args:
            test_data_path: Chemin vers les données de test
            confidence_threshold: Seuil de confiance

        Returns:
            Dictionnaire avec les résultats d'évaluation
        """
        logger.info(f"🧪 Évaluation sur le jeu de test: {test_data_path}")

        # Utiliser la fonction de validation intégrée de YOLOv8
        if hasattr(self.model, "val"):
            results = self.model.val(
                data=test_data_path,
                conf=confidence_threshold,
                save=False,
                verbose=False,
            )

            # Extraire les métriques
            metrics = {
                "precision": (
                    float(results.box.mp) if hasattr(results.box, "mp") else 0.0
                ),
                "recall": float(results.box.mr) if hasattr(results.box, "mr") else 0.0,
                "mAP50": (
                    float(results.box.map50) if hasattr(results.box, "map50") else 0.0
                ),
                "mAP50_95": (
                    float(results.box.map) if hasattr(results.box, "map") else 0.0
                ),
                "f1_score": 0.0,  # Calculé séparément
                "confidence_threshold": confidence_threshold,
            }

            # Calculer F1-Score
            if metrics["precision"] > 0 and metrics["recall"] > 0:
                metrics["f1_score"] = (
                    2
                    * (metrics["precision"] * metrics["recall"])
                    / (metrics["precision"] + metrics["recall"])
                )

            logger.info(f"📊 Résultats d'évaluation:")
            logger.info(f"   • Précision: {metrics['precision']:.3f}")
            logger.info(f"   • Rappel: {metrics['recall']:.3f}")
            logger.info(f"   • F1-Score: {metrics['f1_score']:.3f}")
            logger.info(f"   • mAP@0.5: {metrics['mAP50']:.3f}")
            logger.info(f"   • mAP@0.5:0.95: {metrics['mAP50_95']:.3f}")

            return metrics

        else:
            logger.error("❌ Fonction de validation non disponible")
            return {}

    def analyze_predictions(
        self, test_images_dir: str, output_dir: str = "evaluation_results"
    ) -> Dict[str, Any]:
        """
        Analyse détaillée des prédictions sur des images de test.

        Args:
            test_images_dir: Répertoire contenant les images de test
            output_dir: Répertoire de sortie pour les résultats

        Returns:
            Analyse détaillée des prédictions
        """
        logger.info(f"🔍 Analyse des prédictions: {test_images_dir}")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Trouver toutes les images de test
        test_images = list(Path(test_images_dir).glob("*.jpg")) + list(
            Path(test_images_dir).glob("*.png")
        )

        if not test_images:
            logger.warning(f"Aucune image trouvée dans {test_images_dir}")
            return {}

        logger.info(f"📸 Analyse de {len(test_images)} images...")

        # Effectuer les prédictions
        results = self.model.predict(
            source=test_images_dir, conf=0.25, save=False, verbose=False
        )

        # Analyser les résultats
        analysis = {
            "total_images": len(test_images),
            "images_with_detections": 0,
            "total_detections": 0,
            "confidence_scores": [],
            "detection_sizes": [],
            "predictions_per_image": [],
        }

        for i, result in enumerate(results):
            if result.boxes is not None:
                num_detections = len(result.boxes)
                if num_detections > 0:
                    analysis["images_with_detections"] += 1
                    analysis["total_detections"] += num_detections

                    # Extraire les scores de confiance
                    confidences = result.boxes.conf.cpu().numpy()
                    analysis["confidence_scores"].extend(confidences)

                    # Extraire les tailles des détections
                    boxes = result.boxes.xywh.cpu().numpy()  # x, y, w, h
                    for box in boxes:
                        width, height = box[2], box[3]
                        area = width * height
                        analysis["detection_sizes"].append(area)

                analysis["predictions_per_image"].append(num_detections)

        # Calculer des statistiques
        analysis["detection_rate"] = (
            analysis["images_with_detections"] / analysis["total_images"]
        )
        analysis["avg_detections_per_image"] = (
            analysis["total_detections"] / analysis["total_images"]
        )

        if analysis["confidence_scores"]:
            analysis["mean_confidence"] = np.mean(analysis["confidence_scores"])
            analysis["std_confidence"] = np.std(analysis["confidence_scores"])
        else:
            analysis["mean_confidence"] = 0.0
            analysis["std_confidence"] = 0.0

        logger.info(f"📊 Analyse des prédictions:")
        logger.info(
            f"   • Images avec détections: {analysis['images_with_detections']}/{analysis['total_images']} ({analysis['detection_rate']:.1%})"
        )
        logger.info(f"   • Détections totales: {analysis['total_detections']}")
        logger.info(
            f"   • Détections par image (moyenne): {analysis['avg_detections_per_image']:.2f}"
        )
        logger.info(
            f"   • Confiance moyenne: {analysis['mean_confidence']:.3f} ± {analysis['std_confidence']:.3f}"
        )

        # Sauvegarder l'analyse
        analysis_file = output_path / "prediction_analysis.json"
        with open(analysis_file, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, default=str)

        return analysis

    def create_evaluation_report(
        self,
        metrics: Dict[str, Any],
        analysis: Dict[str, Any],
        output_dir: str = "evaluation_results",
    ) -> str:
        """
        Crée un rapport d'évaluation complet.

        Args:
            metrics: Métriques d'évaluation
            analysis: Analyse des prédictions
            output_dir: Répertoire de sortie

        Returns:
            Chemin du rapport généré
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        report_file = output_path / "evaluation_report.md"

        with open(report_file, "w", encoding="utf-8") as f:
            f.write(
                "# 🦴 Rapport d'Évaluation - Détection de Fractures Pédiatriques\n\n"
            )
            f.write("## 📊 Métriques de Performance\n\n")

            if metrics:
                f.write("### 🎯 Métriques Principales\n\n")
                f.write("| Métrique | Valeur |\n")
                f.write("|----------|--------|\n")
                f.write(f"| Précision | {metrics.get('precision', 0):.3f} |\n")
                f.write(f"| Rappel | {metrics.get('recall', 0):.3f} |\n")
                f.write(f"| F1-Score | {metrics.get('f1_score', 0):.3f} |\n")
                f.write(f"| mAP@0.5 | {metrics.get('mAP50', 0):.3f} |\n")
                f.write(f"| mAP@0.5:0.95 | {metrics.get('mAP50_95', 0):.3f} |\n\n")

                # Évaluation de la performance
                precision = metrics.get("precision", 0)
                recall = metrics.get("recall", 0)
                f1 = metrics.get("f1_score", 0)

                f.write("### 📈 Évaluation de la Performance\n\n")

                if precision >= 0.85 and recall >= 0.80:
                    f.write(
                        "✅ **Excellente performance** - Adapté pour usage clinique\n"
                    )
                elif precision >= 0.75 and recall >= 0.70:
                    f.write(
                        "🟡 **Bonne performance** - Nécessite supervision clinique\n"
                    )
                else:
                    f.write(
                        "🔴 **Performance insuffisante** - Amélioration nécessaire\n"
                    )

                f.write("\n")

            if analysis:
                f.write("## 🔍 Analyse des Prédictions\n\n")
                f.write(f"- **Images analysées**: {analysis.get('total_images', 0)}\n")
                f.write(
                    f"- **Images avec détections**: {analysis.get('images_with_detections', 0)} ({analysis.get('detection_rate', 0):.1%})\n"
                )
                f.write(
                    f"- **Détections totales**: {analysis.get('total_detections', 0)}\n"
                )
                f.write(
                    f"- **Détections par image (moyenne)**: {analysis.get('avg_detections_per_image', 0):.2f}\n"
                )
                f.write(
                    f"- **Confiance moyenne**: {analysis.get('mean_confidence', 0):.3f} ± {analysis.get('std_confidence', 0):.3f}\n\n"
                )

            f.write("## ⚠️ Recommandations Cliniques\n\n")
            f.write("1. **Usage prévu**: Outil d'aide au diagnostic uniquement\n")
            f.write("2. **Supervision**: Toujours valider par un radiologue qualifié\n")
            f.write(
                "3. **Seuils**: Ajuster le seuil de confiance selon le contexte clinique\n"
            )
            f.write(
                "4. **Formations**: Former le personnel à l'interprétation des résultats\n\n"
            )

            f.write("## 📞 Contact\n\n")
            f.write(
                "Pour questions techniques ou cliniques, contacter l'équipe de développement.\n"
            )

        logger.info(f"📄 Rapport d'évaluation généré: {report_file}")
        return str(report_file)

    def plot_evaluation_metrics(
        self, metrics: Dict[str, Any], analysis: Dict[str, Any], save_path: str = None
    ) -> plt.Figure:
        """
        Crée des visualisations des métriques d'évaluation.

        Args:
            metrics: Métriques d'évaluation
            analysis: Analyse des prédictions
            save_path: Chemin de sauvegarde (optionnel)

        Returns:
            Figure matplotlib
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            "📊 Évaluation du Modèle de Détection de Fractures",
            fontsize=16,
            fontweight="bold",
        )

        # 1. Métriques principales (barres)
        if metrics:
            metric_names = ["Précision", "Rappel", "F1-Score", "mAP@0.5"]
            metric_values = [
                metrics.get("precision", 0),
                metrics.get("recall", 0),
                metrics.get("f1_score", 0),
                metrics.get("mAP50", 0),
            ]

            colors = ["skyblue", "lightcoral", "lightgreen", "gold"]
            bars = axes[0, 0].bar(metric_names, metric_values, color=colors, alpha=0.7)
            axes[0, 0].set_title("🎯 Métriques Principales")
            axes[0, 0].set_ylim([0, 1])
            axes[0, 0].set_ylabel("Score")

            # Ajouter les valeurs sur les barres
            for bar, value in zip(bars, metric_values):
                height = bar.get_height()
                axes[0, 0].text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

        # 2. Distribution des confidences
        if analysis and analysis.get("confidence_scores"):
            axes[0, 1].hist(
                analysis["confidence_scores"],
                bins=20,
                alpha=0.7,
                color="orange",
                edgecolor="black",
            )
            axes[0, 1].set_title("📈 Distribution des Scores de Confiance")
            axes[0, 1].set_xlabel("Score de Confiance")
            axes[0, 1].set_ylabel("Nombre de Détections")
            axes[0, 1].grid(True, alpha=0.3)

        # 3. Détections par image
        if analysis and analysis.get("predictions_per_image"):
            pred_counts = np.bincount(analysis["predictions_per_image"])
            axes[1, 0].bar(
                range(len(pred_counts)), pred_counts, alpha=0.7, color="purple"
            )
            axes[1, 0].set_title("📊 Détections par Image")
            axes[1, 0].set_xlabel("Nombre de Détections")
            axes[1, 0].set_ylabel("Nombre d'Images")
            axes[1, 0].grid(True, alpha=0.3)

        # 4. Résumé textuel
        summary_text = "Résumé de l'Évaluation:\n\n"
        if metrics:
            summary_text += f"• Précision: {metrics.get('precision', 0):.3f}\n"
            summary_text += f"• Rappel: {metrics.get('recall', 0):.3f}\n"
            summary_text += f"• F1-Score: {metrics.get('f1_score', 0):.3f}\n"
            summary_text += f"• mAP@0.5: {metrics.get('mAP50', 0):.3f}\n\n"

        if analysis:
            summary_text += f"• Images analysées: {analysis.get('total_images', 0)}\n"
            summary_text += (
                f"• Taux de détection: {analysis.get('detection_rate', 0):.1%}\n"
            )
            summary_text += (
                f"• Confiance moy.: {analysis.get('mean_confidence', 0):.3f}\n"
            )

        axes[1, 1].text(
            0.05,
            0.95,
            summary_text,
            transform=axes[1, 1].transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
        )
        axes[1, 1].set_xlim([0, 1])
        axes[1, 1].set_ylim([0, 1])
        axes[1, 1].axis("off")
        axes[1, 1].set_title("📋 Résumé")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"📊 Graphiques sauvés: {save_path}")

        return fig


def parse_arguments():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="🔍 Évaluation du modèle de détection de fractures pédiatriques"
    )

    parser.add_argument(
        "--model", type=str, required=True, help="Chemin vers le modèle entraîné (.pt)"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        required=True,
        help="Chemin vers les données de test (dataset.yaml ou répertoire images)",
    )
    parser.add_argument(
        "--output", type=str, default="evaluation_results", help="Répertoire de sortie"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.25, help="Seuil de confiance"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/model_config.yaml",
        help="Configuration du modèle",
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

    logger.info("🔍 Évaluation du modèle de détection de fractures pédiatriques")

    try:
        # Initialiser l'évaluateur
        evaluator = ModelEvaluator(args.model, args.config)

        # Créer le répertoire de sortie
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)

        # Évaluer sur le jeu de test
        metrics = {}
        if Path(args.test_data).suffix == ".yaml":
            # Fichier de configuration YOLO
            metrics = evaluator.evaluate_on_test_set(args.test_data, args.confidence)

        # Analyser les prédictions
        analysis = {}
        if Path(args.test_data).is_dir():
            # Répertoire d'images
            analysis = evaluator.analyze_predictions(args.test_data, args.output)

        # Créer les visualisations
        if metrics or analysis:
            plot_path = output_path / "evaluation_plots.png"
            evaluator.plot_evaluation_metrics(metrics, analysis, str(plot_path))

        # Générer le rapport
        if metrics or analysis:
            report_path = evaluator.create_evaluation_report(
                metrics, analysis, args.output
            )
            logger.info(f"📄 Rapport complet: {report_path}")

        logger.info("✅ Évaluation terminée avec succès!")

    except Exception as e:
        logger.error(f"❌ Erreur lors de l'évaluation: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
