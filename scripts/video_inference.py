"""
Script pour inférence vidéo et webcam en temps réel avec YOLOv8.
"""

import argparse
import cv2
import time
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Any

# Ajouter le répertoire parent au path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ Ultralytics non installé. Installez avec: pip install ultralytics")
    sys.exit(1)

from utils.visualization import draw_detections, create_confidence_plot
from utils.model_utils import load_yolo_model
import yaml


class VideoInference:
    """Classe pour l'inférence vidéo en temps réel."""

    def __init__(self, model_path: str, config_path: str = None):
        """
        Initialise l'inférence vidéo.

        Args:
            model_path: Chemin vers le modèle YOLOv8
            config_path: Chemin vers le fichier de configuration
        """
        self.model_path = model_path
        self.model = None
        self.config = self._load_config(config_path)

        # Paramètres par défaut
        self.conf_threshold = self.config.get("confidence_threshold", 0.25)
        self.iou_threshold = self.config.get("iou_threshold", 0.45)
        self.class_names = self.config.get("class_names", ["fracture"])

        # Statistiques
        self.frame_count = 0
        self.detection_count = 0
        self.fps_history = []

        print(f"🎥 Initialisation inférence vidéo")
        print(f"   • Modèle: {model_path}")
        print(f"   • Seuil confiance: {self.conf_threshold}")
        print(f"   • Seuil IoU: {self.iou_threshold}")
        print(f"   • Classes: {self.class_names}")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Charge la configuration depuis un fichier YAML."""
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f)
            except Exception as e:
                print(f"⚠️  Erreur chargement config: {e}")

        # Configuration par défaut
        return {
            "confidence_threshold": 0.25,
            "iou_threshold": 0.45,
            "class_names": ["fracture"],
            "display_labels": True,
            "display_confidence": True,
            "display_fps": True,
            "save_detections": False,
        }

    def load_model(self):
        """Charge le modèle YOLOv8."""
        try:
            self.model = YOLO(self.model_path)
            print(f"✅ Modèle YOLOv8 chargé depuis: {self.model_path}")
            return True
        except Exception as e:
            print(f"❌ Erreur chargement modèle: {e}")
            return False

    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Traite une frame vidéo et retourne les détections.

        Args:
            frame: Frame vidéo en format numpy array

        Returns:
            frame_annotated, detections, inference_time
        """
        if self.model is None:
            return frame, [], 0

        start_time = time.time()

        try:
            # Inférence YOLOv8
            results = self.model(
                frame, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False
            )

            inference_time = time.time() - start_time

            # Extraire les détections
            detections = []
            frame_annotated = frame.copy()

            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        detection = {
                            "confidence": float(box.conf),
                            "class_id": int(box.cls),
                            "class_name": (
                                self.class_names[int(box.cls)]
                                if int(box.cls) < len(self.class_names)
                                else "unknown"
                            ),
                            "bbox": box.xyxy[0].cpu().numpy(),  # [x1, y1, x2, y2]
                        }
                        detections.append(detection)

            # Annoter la frame
            if detections and self.config.get("display_labels", True):
                frame_annotated = self._annotate_frame(frame_annotated, detections)

            self.detection_count += len(detections)

            return frame_annotated, detections, inference_time

        except Exception as e:
            print(f"❌ Erreur traitement frame: {e}")
            return frame, [], 0

    def _annotate_frame(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Annote la frame avec les détections."""
        for detection in detections:
            bbox = detection["bbox"].astype(int)
            x1, y1, x2, y2 = bbox
            confidence = detection["confidence"]
            class_name = detection["class_name"]

            # Couleur selon la classe
            color = (0, 255, 0) if class_name == "fracture" else (255, 0, 0)

            # Dessiner la bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label avec confiance
            if self.config.get("display_confidence", True):
                label = f"{class_name}: {confidence:.2f}"
            else:
                label = class_name

            # Fond pour le texte
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                frame, (x1, y1 - label_height - 10), (x1 + label_width, y1), color, -1
            )

            # Texte
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        return frame

    def process_webcam(self, camera_id: int = 0, display: bool = True):
        """
        Traite le flux webcam en temps réel.

        Args:
            camera_id: ID de la caméra (0 par défaut)
            display: Afficher la vidéo en temps réel
        """
        if not self.load_model():
            return False

        print(f"📹 Ouverture webcam (ID: {camera_id})...")
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print(f"❌ Impossible d'ouvrir la webcam {camera_id}")
            return False

        # Paramètres de la webcam
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        print(f"✅ Webcam active - Appuyez sur 'q' pour quitter")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Erreur lecture frame")
                    break

                # Traitement de la frame
                start_frame = time.time()
                frame_annotated, detections, inference_time = self.process_frame(frame)
                frame_time = time.time() - start_frame

                # Calculer FPS
                fps = 1.0 / frame_time if frame_time > 0 else 0
                self.fps_history.append(fps)
                if len(self.fps_history) > 30:  # Garder 30 dernières mesures
                    self.fps_history.pop(0)

                avg_fps = np.mean(self.fps_history)

                # Afficher les statistiques
                if self.config.get("display_fps", True):
                    stats_text = [
                        f"FPS: {avg_fps:.1f}",
                        f"Inference: {inference_time*1000:.1f}ms",
                        f"Detections: {len(detections)}",
                        f"Frame: {self.frame_count}",
                    ]

                    for i, text in enumerate(stats_text):
                        cv2.putText(
                            frame_annotated,
                            text,
                            (10, 30 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2,
                        )

                # Affichage
                if display:
                    cv2.imshow("Détection de Fractures - Webcam", frame_annotated)

                    # Contrôles clavier
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord("s"):
                        # Sauvegarder la frame actuelle
                        filename = f"detection_frame_{int(time.time())}.jpg"
                        cv2.imwrite(filename, frame_annotated)
                        print(f"💾 Frame sauvée: {filename}")

                self.frame_count += 1

                # Affichage périodique des statistiques
                if self.frame_count % 100 == 0:
                    print(
                        f"📊 Frame {self.frame_count}: {avg_fps:.1f} FPS, {self.detection_count} détections totales"
                    )

        except KeyboardInterrupt:
            print(f"\n⏹️  Arrêt demandé par l'utilisateur")

        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()

            # Statistiques finales
            print(f"\n📊 Statistiques finales:")
            print(f"   • Frames traitées: {self.frame_count}")
            print(f"   • Détections totales: {self.detection_count}")
            print(f"   • FPS moyen: {np.mean(self.fps_history):.1f}")
            print(
                f"   • Détections par frame: {self.detection_count/self.frame_count:.2f}"
            )

        return True

    def process_video_file(
        self, video_path: str, output_path: str = None, display: bool = True
    ):
        """
        Traite un fichier vidéo.

        Args:
            video_path: Chemin vers la vidéo d'entrée
            output_path: Chemin pour sauvegarder la vidéo annotée (optionnel)
            display: Afficher la vidéo pendant le traitement
        """
        if not self.load_model():
            return False

        video_path = Path(video_path)
        if not video_path.exists():
            print(f"❌ Fichier vidéo non trouvé: {video_path}")
            return False

        print(f"🎥 Traitement vidéo: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Impossible d'ouvrir la vidéo: {video_path}")
            return False

        # Propriétés de la vidéo
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(
            f"📊 Propriétés vidéo: {width}x{height}, {fps} FPS, {total_frames} frames"
        )

        # Préparer l'écriture si nécessaire
        writer = None
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            print(f"💾 Sortie vidéo: {output_path}")

        try:
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Traitement de la frame
                frame_annotated, detections, inference_time = self.process_frame(frame)

                # Sauvegarder la frame annotée
                if writer:
                    writer.write(frame_annotated)

                # Affichage optionnel
                if display:
                    # Redimensionner pour l'affichage si nécessaire
                    display_frame = frame_annotated
                    if width > 1280:
                        scale = 1280 / width
                        new_width = 1280
                        new_height = int(height * scale)
                        display_frame = cv2.resize(
                            frame_annotated, (new_width, new_height)
                        )

                    cv2.imshow("Traitement vidéo", display_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break

                # Progression
                frame_idx += 1
                if frame_idx % (total_frames // 20) == 0:  # Afficher tous les 5%
                    progress = (frame_idx / total_frames) * 100
                    print(
                        f"⏳ Progression: {progress:.1f}% ({frame_idx}/{total_frames})"
                    )

        except KeyboardInterrupt:
            print(f"\n⏹️  Arrêt demandé par l'utilisateur")

        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()

            print(f"✅ Traitement terminé: {frame_idx} frames traitées")
            if output_path and output_path.exists():
                print(f"💾 Vidéo annotée sauvée: {output_path}")

        return True


def parse_arguments():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="🎥 Inférence vidéo et webcam avec YOLOv8 pour détection de fractures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  # Webcam en temps réel
  python scripts/video_inference.py --webcam --model models/fracture_detection.pt
  
  # Fichier vidéo
  python scripts/video_inference.py --video input.mp4 --model models/fracture_detection.pt --output result.mp4
  
  # Webcam avec configuration personnalisée
  python scripts/video_inference.py --webcam --model models/fracture_detection.pt --config config/model_config.yaml --camera 1
        """,
    )

    # Mode d'entrée
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--webcam",
        action="store_true",
        help="Utiliser la webcam pour l'inférence en temps réel",
    )
    group.add_argument("--video", type=str, help="Fichier vidéo à traiter")

    # Modèle et configuration
    parser.add_argument(
        "--model", type=str, required=True, help="Chemin vers le modèle YOLOv8 (.pt)"
    )
    parser.add_argument("--config", type=str, help="Fichier de configuration YAML")

    # Paramètres
    parser.add_argument(
        "--camera", type=int, default=0, help="ID de la caméra (défaut: 0)"
    )
    parser.add_argument(
        "--output", type=str, help="Fichier de sortie pour vidéo annotée"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25, help="Seuil de confiance (défaut: 0.25)"
    )
    parser.add_argument(
        "--iou", type=float, default=0.45, help="Seuil IoU pour NMS (défaut: 0.45)"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Ne pas afficher la vidéo pendant le traitement",
    )

    return parser.parse_args()


def main():
    """Fonction principale."""
    args = parse_arguments()

    print("🎥 Inférence vidéo YOLOv8 - Détection de fractures")

    # Vérifier que le modèle existe
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Modèle non trouvé: {model_path}")
        return

    # Initialiser l'inférence vidéo
    video_inference = VideoInference(str(model_path), args.config)

    # Paramètres personnalisés
    if args.conf:
        video_inference.conf_threshold = args.conf
    if args.iou:
        video_inference.iou_threshold = args.iou

    # Exécuter selon le mode
    success = False

    if args.webcam:
        print(f"📹 Mode webcam (caméra {args.camera})")
        success = video_inference.process_webcam(
            camera_id=args.camera, display=not args.no_display
        )

    elif args.video:
        print(f"🎥 Mode fichier vidéo: {args.video}")
        success = video_inference.process_video_file(
            video_path=args.video, output_path=args.output, display=not args.no_display
        )

    if success:
        print(f"🎉 Inférence vidéo terminée avec succès!")
    else:
        print(f"❌ Échec de l'inférence vidéo")


if __name__ == "__main__":
    main()
