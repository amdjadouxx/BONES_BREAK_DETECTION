"""
Utilitaires pour le calcul et l'affichage des métriques d'évaluation.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    average_precision_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger


class DetectionMetrics:
    """Calculateur de métriques pour la détection d'objets."""

    def __init__(self, iou_threshold: float = 0.5):
        """
        Initialise le calculateur de métriques.

        Args:
            iou_threshold: Seuil IoU pour considérer une détection comme correcte
        """
        self.iou_threshold = iou_threshold

    def calculate_iou(self, box1: Dict[str, float], box2: Dict[str, float]) -> float:
        """
        Calcule l'Intersection over Union (IoU) entre deux bounding boxes.

        Args:
            box1: Première bounding box {'x1', 'y1', 'x2', 'y2'}
            box2: Deuxième bounding box {'x1', 'y1', 'x2', 'y2'}

        Returns:
            Score IoU entre 0 et 1
        """
        # Coordonnées de l'intersection
        x1 = max(box1["x1"], box2["x1"])
        y1 = max(box1["y1"], box2["y1"])
        x2 = min(box1["x2"], box2["x2"])
        y2 = min(box1["y2"], box2["y2"])

        # Vérifier si il y a intersection
        if x2 <= x1 or y2 <= y1:
            return 0.0

        # Calcul des aires
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1["x2"] - box1["x1"]) * (box1["y2"] - box1["y1"])
        area2 = (box2["x2"] - box2["x1"]) * (box2["y2"] - box2["y1"])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def match_predictions_to_ground_truth(
        self, predictions: List[Dict[str, Any]], ground_truths: List[Dict[str, Any]]
    ) -> Tuple[List[bool], List[float]]:
        """
        Associe les prédictions aux vérités terrain.

        Args:
            predictions: Liste des prédictions
            ground_truths: Liste des vérités terrain

        Returns:
            Tuple (matches, confidences) où matches[i] indique si predictions[i] est correct
        """
        matches = []
        confidences = []
        used_gts = set()

        # Trier les prédictions par confiance décroissante
        sorted_preds = sorted(predictions, key=lambda x: x["confidence"], reverse=True)

        for pred in sorted_preds:
            confidences.append(pred["confidence"])
            pred_box = pred["bbox"]

            # Trouver la meilleure GT correspondante
            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(ground_truths):
                if gt_idx in used_gts:
                    continue

                gt_box = gt["bbox"]
                iou = self.calculate_iou(pred_box, gt_box)

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            # Vérifier si la correspondance est valide
            if best_iou >= self.iou_threshold and best_gt_idx != -1:
                matches.append(True)
                used_gts.add(best_gt_idx)
            else:
                matches.append(False)

        return matches, confidences

    def calculate_precision_recall_at_iou(
        self,
        all_predictions: List[List[Dict[str, Any]]],
        all_ground_truths: List[List[Dict[str, Any]]],
    ) -> Dict[str, float]:
        """
        Calcule précision, rappel et F1-score pour un seuil IoU donné.

        Args:
            all_predictions: Prédictions pour chaque image
            all_ground_truths: Vérités terrain pour chaque image

        Returns:
            Dictionnaire avec les métriques
        """
        all_matches = []
        all_confidences = []
        total_ground_truths = 0

        # Traiter chaque image
        for predictions, ground_truths in zip(all_predictions, all_ground_truths):
            matches, confidences = self.match_predictions_to_ground_truth(
                predictions, ground_truths
            )
            all_matches.extend(matches)
            all_confidences.extend(confidences)
            total_ground_truths += len(ground_truths)

        if not all_matches:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "average_precision": 0.0,
                "total_predictions": 0,
                "total_ground_truths": total_ground_truths,
            }

        # Calcul des métriques
        true_positives = sum(all_matches)
        false_positives = len(all_matches) - true_positives
        false_negatives = total_ground_truths - true_positives

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0.0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0.0
        )
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # Average Precision (AP)
        if len(set(all_matches)) > 1:  # Si on a à la fois True et False
            ap = average_precision_score(all_matches, all_confidences)
        else:
            ap = 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "average_precision": ap,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "total_predictions": len(all_matches),
            "total_ground_truths": total_ground_truths,
            "iou_threshold": self.iou_threshold,
        }

    def calculate_map_at_multiple_ious(
        self,
        all_predictions: List[List[Dict[str, Any]]],
        all_ground_truths: List[List[Dict[str, Any]]],
        iou_thresholds: List[float] = None,
    ) -> Dict[str, float]:
        """
        Calcule le mAP pour plusieurs seuils IoU.

        Args:
            all_predictions: Prédictions pour chaque image
            all_ground_truths: Vérités terrain pour chaque image
            iou_thresholds: Liste des seuils IoU

        Returns:
            Dictionnaire avec mAP à différents seuils
        """
        if iou_thresholds is None:
            iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

        aps = []

        for iou_thresh in iou_thresholds:
            original_threshold = self.iou_threshold
            self.iou_threshold = iou_thresh

            metrics = self.calculate_precision_recall_at_iou(
                all_predictions, all_ground_truths
            )
            aps.append(metrics["average_precision"])

            self.iou_threshold = original_threshold

        return {
            "mAP50": aps[0] if iou_thresholds[0] == 0.5 else 0.0,
            "mAP50-95": np.mean(aps),
            "individual_aps": dict(zip(iou_thresholds, aps)),
        }


class ClassificationMetrics:
    """Calculateur de métriques pour la classification."""

    def __init__(self):
        """Initialise le calculateur."""
        pass

    def calculate_metrics(
        self, y_true: List[int], y_pred: List[int], y_scores: List[float] = None
    ) -> Dict[str, Any]:
        """
        Calcule les métriques de classification.

        Args:
            y_true: Vraies étiquettes
            y_pred: Prédictions
            y_scores: Scores de probabilité (optionnel)

        Returns:
            Dictionnaire avec toutes les métriques
        """
        metrics = {}

        # Métriques de base
        metrics["accuracy"] = np.mean(np.array(y_true) == np.array(y_pred))
        metrics["precision"] = precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
        metrics["recall"] = recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
        metrics["f1_score"] = f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        )

        # Matrice de confusion
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

        # Rapport détaillé
        metrics["classification_report"] = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )

        # Métriques avec scores si disponibles
        if y_scores is not None:
            try:
                # Pour classification binaire
                if len(set(y_true)) == 2:
                    metrics["roc_auc"] = roc_auc_score(y_true, y_scores)
                    fpr, tpr, _ = roc_curve(y_true, y_scores)
                    metrics["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

                    precision, recall, _ = precision_recall_curve(y_true, y_scores)
                    metrics["pr_curve"] = {
                        "precision": precision.tolist(),
                        "recall": recall.tolist(),
                    }

                metrics["average_precision"] = average_precision_score(y_true, y_scores)

            except Exception as e:
                logger.warning(f"Erreur calcul métriques avancées: {e}")

        return metrics

    def plot_confusion_matrix(
        self,
        y_true: List[int],
        y_pred: List[int],
        class_names: List[str] = None,
        figsize: Tuple[int, int] = (8, 6),
    ) -> plt.Figure:
        """
        Affiche la matrice de confusion.

        Args:
            y_true: Vraies étiquettes
            y_pred: Prédictions
            class_names: Noms des classes
            figsize: Taille de la figure

        Returns:
            Figure matplotlib
        """
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=figsize)

        # Normaliser pour obtenir des pourcentages
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2%",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )

        ax.set_title("Matrice de Confusion Normalisée")
        ax.set_ylabel("Vraie Étiquette")
        ax.set_xlabel("Prédiction")

        plt.tight_layout()
        return fig

    def plot_roc_curve(
        self,
        y_true: List[int],
        y_scores: List[float],
        figsize: Tuple[int, int] = (8, 6),
    ) -> plt.Figure:
        """
        Affiche la courbe ROC.

        Args:
            y_true: Vraies étiquettes
            y_scores: Scores de probabilité
            figsize: Taille de la figure

        Returns:
            Figure matplotlib
        """
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)

        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(
            fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc:.3f})"
        )
        ax.plot(
            [0, 1],
            [0, 1],
            color="navy",
            lw=2,
            linestyle="--",
            label="Random classifier",
        )

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Taux de Faux Positifs")
        ax.set_ylabel("Taux de Vrais Positifs")
        ax.set_title("Courbe ROC (Receiver Operating Characteristic)")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_precision_recall_curve(
        self,
        y_true: List[int],
        y_scores: List[float],
        figsize: Tuple[int, int] = (8, 6),
    ) -> plt.Figure:
        """
        Affiche la courbe Precision-Recall.

        Args:
            y_true: Vraies étiquettes
            y_scores: Scores de probabilité
            figsize: Taille de la figure

        Returns:
            Figure matplotlib
        """
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)

        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(
            recall, precision, color="darkred", lw=2, label=f"PR curve (AP = {ap:.3f})"
        )

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Rappel")
        ax.set_ylabel("Précision")
        ax.set_title("Courbe Précision-Rappel")
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


def create_metrics_dashboard(
    detection_metrics: Dict[str, Any], classification_metrics: Dict[str, Any] = None
) -> plt.Figure:
    """
    Crée un dashboard complet des métriques.

    Args:
        detection_metrics: Métriques de détection
        classification_metrics: Métriques de classification (optionnel)

    Returns:
        Figure matplotlib avec le dashboard
    """
    if classification_metrics is None:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    else:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes = axes.flatten()

    # 1. Métriques principales (barres)
    metrics_names = ["Precision", "Recall", "F1-Score", "mAP@0.5"]
    metrics_values = [
        detection_metrics.get("precision", 0),
        detection_metrics.get("recall", 0),
        detection_metrics.get("f1_score", 0),
        detection_metrics.get("mAP50", 0),
    ]

    axes[0].bar(
        metrics_names,
        metrics_values,
        color=["skyblue", "lightcoral", "lightgreen", "gold"],
    )
    axes[0].set_title("Métriques Principales")
    axes[0].set_ylim([0, 1])
    axes[0].set_ylabel("Score")
    for i, v in enumerate(metrics_values):
        axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center")

    # 2. Répartition TP/FP/FN
    tp = detection_metrics.get("true_positives", 0)
    fp = detection_metrics.get("false_positives", 0)
    fn = detection_metrics.get("false_negatives", 0)

    axes[1].pie(
        [tp, fp, fn],
        labels=["Vrais Positifs", "Faux Positifs", "Faux Négatifs"],
        colors=["lightgreen", "lightcoral", "lightyellow"],
        autopct="%1.1f%%",
    )
    axes[1].set_title("Répartition des Prédictions")

    # 3. Comparaison des seuils IoU (si disponible)
    if "individual_aps" in detection_metrics:
        iou_thresholds = list(detection_metrics["individual_aps"].keys())
        aps = list(detection_metrics["individual_aps"].values())

        axes[2].plot(iou_thresholds, aps, "o-", color="purple")
        axes[2].set_title("AP vs Seuil IoU")
        axes[2].set_xlabel("Seuil IoU")
        axes[2].set_ylabel("Average Precision")
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(
            0.5,
            0.5,
            "Pas de données\npour les seuils IoU",
            ha="center",
            va="center",
            transform=axes[2].transAxes,
        )
        axes[2].set_title("AP vs Seuil IoU")

    # 4. Résumé textuel
    summary_text = f"""
Résumé des Performances:

• Total prédictions: {detection_metrics.get('total_predictions', 0)}
• Total objets réels: {detection_metrics.get('total_ground_truths', 0)}
• Précision: {detection_metrics.get('precision', 0):.3f}
• Rappel: {detection_metrics.get('recall', 0):.3f}
• F1-Score: {detection_metrics.get('f1_score', 0):.3f}

Seuil IoU utilisé: {detection_metrics.get('iou_threshold', 0.5)}
    """

    axes[3].text(
        0.05,
        0.95,
        summary_text,
        transform=axes[3].transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
    )
    axes[3].set_xlim([0, 1])
    axes[3].set_ylim([0, 1])
    axes[3].axis("off")
    axes[3].set_title("Résumé")

    # Si métriques de classification disponibles
    if classification_metrics and len(axes) > 4:
        # 5. Matrice de confusion
        if "confusion_matrix" in classification_metrics:
            cm = np.array(classification_metrics["confusion_matrix"])
            im = axes[4].imshow(cm, interpolation="nearest", cmap="Blues")
            axes[4].set_title("Matrice de Confusion")

            # Ajouter les valeurs dans les cellules
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    axes[4].text(j, i, str(cm[i, j]), ha="center", va="center")

        # 6. Métriques par classe
        if "classification_report" in classification_metrics:
            report = classification_metrics["classification_report"]
            classes = [
                k
                for k in report.keys()
                if k not in ["accuracy", "macro avg", "weighted avg"]
            ]

            if classes:
                precisions = [report[c]["precision"] for c in classes]
                recalls = [report[c]["recall"] for c in classes]

                x = np.arange(len(classes))
                width = 0.35

                axes[5].bar(x - width / 2, precisions, width, label="Precision")
                axes[5].bar(x + width / 2, recalls, width, label="Recall")
                axes[5].set_title("Métriques par Classe")
                axes[5].set_xticks(x)
                axes[5].set_xticklabels(classes)
                axes[5].legend()

    plt.tight_layout()
    return fig


def save_metrics_report(metrics: Dict[str, Any], output_path: str):
    """
    Sauvegarde un rapport des métriques en format texte.

    Args:
        metrics: Dictionnaire des métriques
        output_path: Chemin de sauvegarde
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("RAPPORT D'ÉVALUATION DES PERFORMANCES\n")
        f.write("=" * 50 + "\n\n")

        # Métriques de détection
        if "precision" in metrics:
            f.write("MÉTRIQUES DE DÉTECTION:\n")
            f.write("-" * 25 + "\n")
            f.write(f"Précision: {metrics.get('precision', 0):.4f}\n")
            f.write(f"Rappel: {metrics.get('recall', 0):.4f}\n")
            f.write(f"F1-Score: {metrics.get('f1_score', 0):.4f}\n")
            f.write(f"Average Precision: {metrics.get('average_precision', 0):.4f}\n")
            f.write(f"mAP@0.5: {metrics.get('mAP50', 0):.4f}\n")
            f.write(f"mAP@0.5:0.95: {metrics.get('mAP50-95', 0):.4f}\n\n")

            f.write(f"Vrais Positifs: {metrics.get('true_positives', 0)}\n")
            f.write(f"Faux Positifs: {metrics.get('false_positives', 0)}\n")
            f.write(f"Faux Négatifs: {metrics.get('false_negatives', 0)}\n\n")

        # Métriques de classification si disponibles
        if "classification_report" in metrics:
            f.write("RAPPORT DE CLASSIFICATION:\n")
            f.write("-" * 30 + "\n")
            report = metrics["classification_report"]

            for class_name, class_metrics in report.items():
                if isinstance(class_metrics, dict):
                    f.write(f"\nClasse '{class_name}':\n")
                    f.write(f"  Précision: {class_metrics.get('precision', 0):.4f}\n")
                    f.write(f"  Rappel: {class_metrics.get('recall', 0):.4f}\n")
                    f.write(f"  F1-Score: {class_metrics.get('f1-score', 0):.4f}\n")
                    f.write(f"  Support: {class_metrics.get('support', 0)}\n")

        f.write(f"\nRapport généré automatiquement.\n")

    logger.info(f"Rapport des métriques sauvé: {output_path}")


if __name__ == "__main__":
    # Test des fonctionnalités
    detection_metrics = DetectionMetrics(iou_threshold=0.5)
    classification_metrics = ClassificationMetrics()

    print("✅ Module metrics initialisé avec succès!")
    print(f"📊 Seuil IoU: {detection_metrics.iou_threshold}")
    print(f"🎯 Prêt pour l'évaluation des performances")
