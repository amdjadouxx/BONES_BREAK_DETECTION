"""
Utilitaires pour la visualisation des résultats et des données médicales.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from loguru import logger


class MedicalImageVisualizer:
    """Visualiseur pour les images médicales et les prédictions."""

    def __init__(self):
        """Initialise le visualiseur."""
        # Couleurs pour les classes (BGR pour OpenCV)
        self.class_colors = {
            0: (0, 0, 255),  # Rouge pour fracture
            1: (0, 255, 0),  # Vert pour normal
            2: (255, 0, 0),  # Bleu pour incertain
        }

        self.class_names = {0: "Fracture", 1: "Normal", 2: "Incertain"}

        # Style matplotlib
        plt.style.use("default")
        sns.set_palette("husl")

    def draw_bboxes_on_image(
        self,
        image_path: str,
        detections: List[Dict[str, Any]],
        output_path: str = None,
        show_confidence: bool = True,
    ) -> np.ndarray:
        """
        Dessine les bounding boxes sur une image.

        Args:
            image_path: Chemin vers l'image
            detections: Liste des détections avec bboxes
            output_path: Chemin de sauvegarde (optionnel)
            show_confidence: Afficher les scores de confiance

        Returns:
            Image avec les bounding boxes
        """
        try:
            # Charger l'image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Impossible de charger l'image: {image_path}")

            # Dessiner chaque détection
            for detection in detections:
                bbox = detection["bbox"]
                confidence = detection["confidence"]
                class_id = detection.get("class_id", 0)
                class_name = detection.get("class_name", "fracture")

                # Coordonnées du rectangle
                x1, y1 = int(bbox["x1"]), int(bbox["y1"])
                x2, y2 = int(bbox["x2"]), int(bbox["y2"])

                # Couleur selon la classe
                color = self.class_colors.get(class_id, (0, 255, 255))

                # Dessiner le rectangle
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

                # Préparer le texte
                if show_confidence:
                    label = f"{class_name}: {confidence:.2f}"
                else:
                    label = class_name

                # Calculer la taille du texte
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, font, font_scale, thickness
                )

                # Dessiner le fond du texte
                cv2.rectangle(
                    image, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1
                )

                # Dessiner le texte
                cv2.putText(
                    image,
                    label,
                    (x1, y1 - 5),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                )

            # Sauvegarder si chemin fourni
            if output_path:
                cv2.imwrite(output_path, image)
                logger.info(f"Image avec détections sauvée: {output_path}")

            return image

        except Exception as e:
            logger.error(f"Erreur visualisation détections: {e}")
            raise

    def create_detection_grid(
        self,
        image_paths: List[str],
        detections_list: List[List[Dict[str, Any]]],
        grid_size: Tuple[int, int] = (3, 3),
        figsize: Tuple[int, int] = (15, 15),
    ) -> plt.Figure:
        """
        Crée une grille d'images avec leurs détections.

        Args:
            image_paths: Liste des chemins d'images
            detections_list: Liste des détections pour chaque image
            grid_size: Taille de la grille (rows, cols)
            figsize: Taille de la figure

        Returns:
            Figure matplotlib
        """
        rows, cols = grid_size
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        for i, (image_path, detections) in enumerate(
            zip(image_paths[: len(axes)], detections_list[: len(axes)])
        ):
            if i >= len(axes):
                break

            try:
                # Charger et traiter l'image
                image = cv2.imread(image_path)
                if image is not None:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image_with_boxes = self.draw_bboxes_cv2(image_rgb, detections)

                    axes[i].imshow(image_with_boxes)
                    axes[i].set_title(
                        f"{Path(image_path).name}\n{len(detections)} détection(s)"
                    )
                else:
                    axes[i].text(
                        0.5, 0.5, "Image\nnon trouvée", ha="center", va="center"
                    )

            except Exception as e:
                axes[i].text(
                    0.5, 0.5, f"Erreur:\n{str(e)[:50]}...", ha="center", va="center"
                )

            axes[i].axis("off")

        # Masquer les axes non utilisés
        for i in range(len(image_paths), len(axes)):
            axes[i].axis("off")

        plt.tight_layout()
        return fig

    def draw_bboxes_cv2(
        self, image: np.ndarray, detections: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Version OpenCV du dessin de bounding boxes."""
        image_copy = image.copy()

        for detection in detections:
            bbox = detection["bbox"]
            confidence = detection["confidence"]
            class_id = detection.get("class_id", 0)
            class_name = detection.get("class_name", "fracture")

            x1, y1 = int(bbox["x1"]), int(bbox["y1"])
            x2, y2 = int(bbox["x2"]), int(bbox["y2"])

            color = self.class_colors.get(class_id, (255, 255, 0))
            # Pour RGB (matplotlib), inverser BGR -> RGB
            color = (color[2], color[1], color[0])

            cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)

            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(
                image_copy,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        return image_copy

    def plot_confidence_distribution(
        self, detections_list: List[List[Dict[str, Any]]]
    ) -> plt.Figure:
        """
        Affiche la distribution des scores de confiance.

        Args:
            detections_list: Liste des détections

        Returns:
            Figure matplotlib
        """
        # Extraire tous les scores de confiance
        confidences = []
        class_names = []

        for detections in detections_list:
            for detection in detections:
                confidences.append(detection["confidence"])
                class_names.append(detection.get("class_name", "fracture"))

        if not confidences:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "Aucune détection trouvée", ha="center", va="center")
            return fig

        # Créer les graphiques
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Histogramme des confidences
        ax1.hist(confidences, bins=20, alpha=0.7, edgecolor="black")
        ax1.set_xlabel("Score de Confiance")
        ax1.set_ylabel("Nombre de Détections")
        ax1.set_title("Distribution des Scores de Confiance")
        ax1.grid(True, alpha=0.3)

        # Box plot par classe
        df = pd.DataFrame({"confidence": confidences, "class": class_names})
        sns.boxplot(data=df, x="class", y="confidence", ax=ax2)
        ax2.set_title("Confiance par Classe")
        ax2.set_ylabel("Score de Confiance")

        plt.tight_layout()
        return fig

    def create_detection_summary_plot(
        self, results_summary: Dict[str, Any]
    ) -> go.Figure:
        """
        Crée un résumé visuel des résultats de détection avec Plotly.

        Args:
            results_summary: Résumé des résultats

        Returns:
            Figure Plotly
        """
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Détections par Image",
                "Distribution des Confidences",
                "Classes Détectées",
                "Métriques Globales",
            ),
            specs=[
                [{"type": "bar"}, {"type": "histogram"}],
                [{"type": "pie"}, {"type": "table"}],
            ],
        )

        # 1. Détections par image
        if "detections_per_image" in results_summary:
            data = results_summary["detections_per_image"]
            fig.add_trace(
                go.Bar(x=list(range(len(data))), y=data, name="Détections"),
                row=1,
                col=1,
            )

        # 2. Distribution des confidences
        if "all_confidences" in results_summary:
            fig.add_trace(
                go.Histogram(
                    x=results_summary["all_confidences"], nbinsx=20, name="Confiance"
                ),
                row=1,
                col=2,
            )

        # 3. Répartition des classes
        if "class_counts" in results_summary:
            labels = list(results_summary["class_counts"].keys())
            values = list(results_summary["class_counts"].values())
            fig.add_trace(
                go.Pie(labels=labels, values=values, name="Classes"), row=2, col=1
            )

        # 4. Tableau des métriques
        if "metrics" in results_summary:
            metrics = results_summary["metrics"]
            fig.add_trace(
                go.Table(
                    header=dict(values=["Métrique", "Valeur"]),
                    cells=dict(
                        values=[
                            list(metrics.keys()),
                            [
                                f"{v:.3f}" if isinstance(v, float) else str(v)
                                for v in metrics.values()
                            ],
                        ]
                    ),
                ),
                row=2,
                col=2,
            )

        fig.update_layout(
            title="Résumé des Détections de Fractures", showlegend=False, height=800
        )

        return fig

    def save_detection_report(
        self, image_path: str, detections: List[Dict[str, Any]], output_dir: str
    ) -> str:
        """
        Sauvegarde un rapport complet de détection.

        Args:
            image_path: Chemin vers l'image
            detections: Liste des détections
            output_dir: Répertoire de sortie

        Returns:
            Chemin du rapport sauvé
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        image_name = Path(image_path).stem

        # 1. Image avec bounding boxes
        image_with_boxes = self.draw_bboxes_on_image(
            image_path, detections, str(output_path / f"{image_name}_detected.jpg")
        )

        # 2. Rapport texte
        report_path = output_path / f"{image_name}_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"RAPPORT DE DÉTECTION - {image_name}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Image source: {image_path}\n")
            f.write(f"Nombre de détections: {len(detections)}\n\n")

            for i, detection in enumerate(detections, 1):
                f.write(f"Détection #{i}:\n")
                f.write(f"  - Classe: {detection.get('class_name', 'fracture')}\n")
                f.write(f"  - Confiance: {detection['confidence']:.3f}\n")
                f.write(
                    f"  - Position: ({detection['bbox']['x1']:.0f}, {detection['bbox']['y1']:.0f}) "
                    f"-> ({detection['bbox']['x2']:.0f}, {detection['bbox']['y2']:.0f})\n"
                )
                f.write(
                    f"  - Taille: {detection['bbox']['width']:.0f} x {detection['bbox']['height']:.0f}\n\n"
                )

        logger.info(f"Rapport sauvé: {report_path}")
        return str(report_path)


def plot_training_metrics(metrics_file: str) -> plt.Figure:
    """
    Affiche les métriques d'entraînement depuis un fichier CSV.

    Args:
        metrics_file: Chemin vers le fichier des métriques

    Returns:
        Figure matplotlib
    """
    try:
        df = pd.read_csv(metrics_file)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss
        if "train_loss" in df.columns:
            axes[0, 0].plot(df["epoch"], df["train_loss"], label="Train Loss")
        if "val_loss" in df.columns:
            axes[0, 0].plot(df["epoch"], df["val_loss"], label="Val Loss")
        axes[0, 0].set_title("Loss Evolution")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Precision/Recall
        if "precision" in df.columns and "recall" in df.columns:
            axes[0, 1].plot(df["epoch"], df["precision"], label="Precision")
            axes[0, 1].plot(df["epoch"], df["recall"], label="Recall")
            axes[0, 1].set_title("Precision & Recall")
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylabel("Score")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # mAP
        if "mAP50" in df.columns:
            axes[1, 0].plot(df["epoch"], df["mAP50"], label="mAP@0.5")
        if "mAP50-95" in df.columns:
            axes[1, 0].plot(df["epoch"], df["mAP50-95"], label="mAP@0.5:0.95")
        axes[1, 0].set_title("Mean Average Precision")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("mAP")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Learning Rate
        if "lr" in df.columns:
            axes[1, 1].plot(df["epoch"], df["lr"])
            axes[1, 1].set_title("Learning Rate")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("Learning Rate")
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    except Exception as e:
        logger.error(f"Erreur chargement métriques: {e}")
        raise


def create_medical_image_montage(
    image_paths: List[str], titles: List[str] = None, grid_size: Tuple[int, int] = None
) -> plt.Figure:
    """
    Crée un montage d'images médicales.

    Args:
        image_paths: Liste des chemins d'images
        titles: Titres pour chaque image
        grid_size: Taille de la grille (auto si None)

    Returns:
        Figure matplotlib
    """
    n_images = len(image_paths)

    if grid_size is None:
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))
    else:
        rows, cols = grid_size

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for i, image_path in enumerate(image_paths[: len(axes)]):
        try:
            image = cv2.imread(image_path)
            if image is not None:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                axes[i].imshow(
                    image_rgb, cmap="gray" if len(image_rgb.shape) == 2 else None
                )

                title = (
                    titles[i] if titles and i < len(titles) else Path(image_path).name
                )
                axes[i].set_title(title)
            else:
                axes[i].text(0.5, 0.5, "Image\nnon trouvée", ha="center", va="center")

        except Exception as e:
            axes[i].text(
                0.5, 0.5, f"Erreur:\n{str(e)[:30]}...", ha="center", va="center"
            )

        axes[i].axis("off")

    # Masquer les axes non utilisés
    for i in range(n_images, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Test des fonctionnalités
    visualizer = MedicalImageVisualizer()

    print("✅ Module visualization initialisé avec succès!")
    print(f"🎨 Classes configurées: {list(visualizer.class_names.values())}")
    print(f"🎯 Prêt pour la visualisation des détections médicales")
