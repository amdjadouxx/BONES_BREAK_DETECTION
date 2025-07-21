"""
Script de conversion des formats d'annotation vers le format YOLO.
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import cv2
from tqdm import tqdm
import sys

# Ajouter le répertoire parent au path
sys.path.append(str(Path(__file__).parent.parent))


class AnnotationConverter:
    """Convertisseur d'annotations vers le format YOLO."""

    def __init__(self, class_mapping: Dict[str, int] = None):
        """
        Initialise le convertisseur.

        Args:
            class_mapping: Dictionnaire de correspondance nom_classe -> id_classe
        """
        self.class_mapping = class_mapping or {"fracture": 0}
        print(f"📋 Classes configurées: {self.class_mapping}")

    def coco_to_yolo(
        self, coco_annotation_file: str, images_dir: str, output_dir: str
    ) -> bool:
        """
        Convertit les annotations COCO vers YOLO.

        Args:
            coco_annotation_file: Fichier JSON des annotations COCO
            images_dir: Répertoire des images
            output_dir: Répertoire de sortie pour les annotations YOLO

        Returns:
            True si succès, False sinon
        """
        try:
            # Charger les annotations COCO
            with open(coco_annotation_file, "r") as f:
                coco_data = json.load(f)

            print(
                f"📥 Chargé {len(coco_data.get('images', []))} images et {len(coco_data.get('annotations', []))} annotations"
            )

            # Créer le répertoire de sortie
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Créer un index des images
            images_index = {img["id"]: img for img in coco_data.get("images", [])}

            # Créer un index des catégories
            categories_index = {
                cat["id"]: cat["name"] for cat in coco_data.get("categories", [])
            }

            # Grouper les annotations par image
            annotations_by_image = {}
            for ann in coco_data.get("annotations", []):
                image_id = ann["image_id"]
                if image_id not in annotations_by_image:
                    annotations_by_image[image_id] = []
                annotations_by_image[image_id].append(ann)

            converted_count = 0

            # Convertir chaque image
            for image_id, image_info in tqdm(
                images_index.items(), desc="Conversion COCO->YOLO"
            ):
                if image_id not in annotations_by_image:
                    continue  # Pas d'annotations pour cette image

                image_width = image_info["width"]
                image_height = image_info["height"]
                image_filename = image_info["file_name"]

                # Créer le fichier d'annotation YOLO correspondant
                yolo_filename = Path(image_filename).stem + ".txt"
                yolo_path = output_path / yolo_filename

                yolo_annotations = []

                for ann in annotations_by_image[image_id]:
                    # Extraire les coordonnées COCO (x, y, width, height)
                    x, y, w, h = ann["bbox"]
                    category_id = ann["category_id"]
                    category_name = categories_index.get(category_id, "unknown")

                    # Convertir le nom de catégorie en ID YOLO
                    if category_name in self.class_mapping:
                        yolo_class_id = self.class_mapping[category_name]
                    else:
                        continue  # Ignorer les classes non mappées

                    # Convertir en format YOLO (coordonnées normalisées, centre)
                    x_center = (x + w / 2) / image_width
                    y_center = (y + h / 2) / image_height
                    width_norm = w / image_width
                    height_norm = h / image_height

                    # S'assurer que les coordonnées sont dans [0, 1]
                    x_center = np.clip(x_center, 0, 1)
                    y_center = np.clip(y_center, 0, 1)
                    width_norm = np.clip(width_norm, 0, 1)
                    height_norm = np.clip(height_norm, 0, 1)

                    yolo_annotations.append(
                        f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}"
                    )

                # Sauvegarder les annotations YOLO
                if yolo_annotations:
                    with open(yolo_path, "w") as f:
                        f.write("\n".join(yolo_annotations))
                    converted_count += 1

            print(
                f"✅ Conversion COCO->YOLO terminée: {converted_count} fichiers créés"
            )
            return True

        except Exception as e:
            print(f"❌ Erreur conversion COCO: {e}")
            return False

    def pascal_voc_to_yolo(
        self, xml_dir: str, images_dir: str, output_dir: str
    ) -> bool:
        """
        Convertit les annotations Pascal VOC (XML) vers YOLO.

        Args:
            xml_dir: Répertoire contenant les fichiers XML
            images_dir: Répertoire des images
            output_dir: Répertoire de sortie pour les annotations YOLO

        Returns:
            True si succès, False sinon
        """
        try:
            xml_files = list(Path(xml_dir).glob("*.xml"))
            if not xml_files:
                print(f"❌ Aucun fichier XML trouvé dans {xml_dir}")
                return False

            print(f"📥 Trouvé {len(xml_files)} fichiers XML à convertir")

            # Créer le répertoire de sortie
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            converted_count = 0

            for xml_file in tqdm(xml_files, desc="Conversion Pascal VOC->YOLO"):
                try:
                    # Parser le fichier XML
                    tree = ET.parse(xml_file)
                    root = tree.getroot()

                    # Obtenir les dimensions de l'image
                    size_elem = root.find("size")
                    if size_elem is None:
                        continue

                    image_width = int(size_elem.find("width").text)
                    image_height = int(size_elem.find("height").text)

                    yolo_annotations = []

                    # Traiter chaque objet
                    for obj in root.findall("object"):
                        # Obtenir le nom de la classe
                        class_name = obj.find("name").text.lower()

                        if class_name not in self.class_mapping:
                            continue  # Ignorer les classes non mappées

                        yolo_class_id = self.class_mapping[class_name]

                        # Obtenir les coordonnées de la bounding box
                        bbox = obj.find("bndbox")
                        xmin = float(bbox.find("xmin").text)
                        ymin = float(bbox.find("ymin").text)
                        xmax = float(bbox.find("xmax").text)
                        ymax = float(bbox.find("ymax").text)

                        # Convertir en format YOLO
                        x_center = (xmin + xmax) / 2 / image_width
                        y_center = (ymin + ymax) / 2 / image_height
                        width = (xmax - xmin) / image_width
                        height = (ymax - ymin) / image_height

                        # S'assurer que les coordonnées sont dans [0, 1]
                        x_center = np.clip(x_center, 0, 1)
                        y_center = np.clip(y_center, 0, 1)
                        width = np.clip(width, 0, 1)
                        height = np.clip(height, 0, 1)

                        yolo_annotations.append(
                            f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                        )

                    # Sauvegarder les annotations YOLO
                    if yolo_annotations:
                        output_file = output_path / f"{xml_file.stem}.txt"
                        with open(output_file, "w") as f:
                            f.write("\n".join(yolo_annotations))
                        converted_count += 1

                except Exception as e:
                    print(f"⚠️  Erreur traitement {xml_file}: {e}")
                    continue

            print(
                f"✅ Conversion Pascal VOC->YOLO terminée: {converted_count} fichiers créés"
            )
            return True

        except Exception as e:
            print(f"❌ Erreur conversion Pascal VOC: {e}")
            return False

    def csv_to_yolo(
        self,
        csv_file: str,
        images_dir: str,
        output_dir: str,
        columns: Dict[str, str] = None,
    ) -> bool:
        """
        Convertit un fichier CSV d'annotations vers YOLO.

        Args:
            csv_file: Fichier CSV des annotations
            images_dir: Répertoire des images
            output_dir: Répertoire de sortie
            columns: Mapping des colonnes CSV {'filename': 'image', 'class': 'label', ...}

        Returns:
            True si succès, False sinon
        """
        # Colonnes par défaut
        default_columns = {
            "filename": "filename",
            "class": "class",
            "x": "x",
            "y": "y",
            "width": "width",
            "height": "height",
            "img_width": "img_width",
            "img_height": "img_height",
        }

        if columns:
            default_columns.update(columns)

        try:
            # Charger le CSV
            df = pd.read_csv(csv_file)
            print(f"📥 Chargé {len(df)} annotations depuis {csv_file}")

            # Vérifier que les colonnes nécessaires existent
            required_cols = ["filename", "class", "x", "y", "width", "height"]
            for col in required_cols:
                if default_columns[col] not in df.columns:
                    print(f"❌ Colonne manquante: {default_columns[col]}")
                    return False

            # Créer le répertoire de sortie
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            converted_files = set()

            # Grouper par fichier image
            for filename, group in tqdm(
                df.groupby(default_columns["filename"]), desc="Conversion CSV->YOLO"
            ):
                try:
                    yolo_annotations = []

                    # Obtenir les dimensions de l'image
                    if (
                        "img_width" in default_columns
                        and default_columns["img_width"] in df.columns
                    ):
                        img_width = group.iloc[0][default_columns["img_width"]]
                        img_height = group.iloc[0][default_columns["img_height"]]
                    else:
                        # Lire les dimensions depuis l'image
                        img_path = Path(images_dir) / filename
                        if img_path.exists():
                            img = cv2.imread(str(img_path))
                            if img is not None:
                                img_height, img_width = img.shape[:2]
                            else:
                                continue
                        else:
                            continue

                    # Convertir chaque annotation
                    for _, row in group.iterrows():
                        class_name = str(row[default_columns["class"]]).lower()

                        if class_name not in self.class_mapping:
                            continue

                        yolo_class_id = self.class_mapping[class_name]

                        # Coordonnées de la bounding box (assumées être x, y, width, height)
                        x = float(row[default_columns["x"]])
                        y = float(row[default_columns["y"]])
                        w = float(row[default_columns["width"]])
                        h = float(row[default_columns["height"]])

                        # Convertir en format YOLO
                        x_center = (x + w / 2) / img_width
                        y_center = (y + h / 2) / img_height
                        width_norm = w / img_width
                        height_norm = h / img_height

                        # S'assurer que les coordonnées sont dans [0, 1]
                        x_center = np.clip(x_center, 0, 1)
                        y_center = np.clip(y_center, 0, 1)
                        width_norm = np.clip(width_norm, 0, 1)
                        height_norm = np.clip(height_norm, 0, 1)

                        yolo_annotations.append(
                            f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}"
                        )

                    # Sauvegarder les annotations YOLO
                    if yolo_annotations:
                        output_file = output_path / f"{Path(filename).stem}.txt"
                        with open(output_file, "w") as f:
                            f.write("\n".join(yolo_annotations))
                        converted_files.add(filename)

                except Exception as e:
                    print(f"⚠️  Erreur traitement {filename}: {e}")
                    continue

            print(
                f"✅ Conversion CSV->YOLO terminée: {len(converted_files)} fichiers créés"
            )
            return True

        except Exception as e:
            print(f"❌ Erreur conversion CSV: {e}")
            return False

    def validate_yolo_annotations(
        self, annotations_dir: str, images_dir: str = None
    ) -> Dict[str, Any]:
        """
        Valide les annotations YOLO.

        Args:
            annotations_dir: Répertoire contenant les annotations YOLO
            images_dir: Répertoire des images (optionnel, pour validation complète)

        Returns:
            Rapport de validation
        """
        annotations_path = Path(annotations_dir)
        report = {
            "total_files": 0,
            "valid_files": 0,
            "invalid_files": [],
            "issues": [],
            "class_distribution": {},
            "statistics": {
                "total_annotations": 0,
                "avg_annotations_per_file": 0,
                "bbox_sizes": [],
            },
        }

        print(f"🔍 Validation des annotations YOLO dans {annotations_dir}")

        # Trouver tous les fichiers d'annotation
        annotation_files = list(annotations_path.glob("*.txt"))
        report["total_files"] = len(annotation_files)

        if not annotation_files:
            print(f"❌ Aucun fichier d'annotation trouvé dans {annotations_dir}")
            return report

        total_annotations = 0

        for ann_file in tqdm(annotation_files, desc="Validation"):
            try:
                valid_file = True
                file_annotations = 0

                with open(ann_file, "r") as f:
                    lines = f.readlines()

                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) != 5:
                        report["issues"].append(
                            f"{ann_file.name}:{line_num} - Format invalide (attendu: class x y w h)"
                        )
                        valid_file = False
                        continue

                    try:
                        class_id = int(parts[0])
                        x, y, w, h = map(float, parts[1:5])

                        # Vérifier que les coordonnées sont dans [0, 1]
                        if not (
                            0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1
                        ):
                            report["issues"].append(
                                f"{ann_file.name}:{line_num} - Coordonnées hors limites [0,1]"
                            )
                            valid_file = False

                        # Vérifier que la classe est valide
                        if class_id < 0:
                            report["issues"].append(
                                f"{ann_file.name}:{line_num} - ID de classe négatif"
                            )
                            valid_file = False

                        # Statistiques
                        report["class_distribution"][class_id] = (
                            report["class_distribution"].get(class_id, 0) + 1
                        )
                        report["statistics"]["bbox_sizes"].append(w * h)
                        file_annotations += 1

                    except ValueError as e:
                        report["issues"].append(
                            f"{ann_file.name}:{line_num} - Erreur parsing: {e}"
                        )
                        valid_file = False

                if valid_file:
                    report["valid_files"] += 1
                else:
                    report["invalid_files"].append(ann_file.name)

                total_annotations += file_annotations

            except Exception as e:
                report["issues"].append(f"{ann_file.name} - Erreur lecture: {e}")
                report["invalid_files"].append(ann_file.name)

        # Calculer les statistiques
        report["statistics"]["total_annotations"] = total_annotations
        if report["total_files"] > 0:
            report["statistics"]["avg_annotations_per_file"] = (
                total_annotations / report["total_files"]
            )

        if report["statistics"]["bbox_sizes"]:
            report["statistics"]["avg_bbox_size"] = np.mean(
                report["statistics"]["bbox_sizes"]
            )
            report["statistics"]["median_bbox_size"] = np.median(
                report["statistics"]["bbox_sizes"]
            )

        # Validation croisée avec les images si disponible
        if images_dir:
            images_path = Path(images_dir)
            image_files = set()
            for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                image_files.update([f.stem for f in images_path.glob(f"*{ext}")])

            annotation_stems = set([f.stem for f in annotation_files])

            missing_annotations = image_files - annotation_stems
            orphan_annotations = annotation_stems - image_files

            if missing_annotations:
                report["issues"].append(
                    f"{len(missing_annotations)} images sans annotations"
                )
            if orphan_annotations:
                report["issues"].append(
                    f"{len(orphan_annotations)} annotations sans images correspondantes"
                )

        # Résumé
        print(f"📊 Résultats de validation:")
        print(f"   • Fichiers totaux: {report['total_files']}")
        print(f"   • Fichiers valides: {report['valid_files']}")
        print(f"   • Fichiers invalides: {len(report['invalid_files'])}")
        print(f"   • Annotations totales: {total_annotations}")
        print(f"   • Problèmes détectés: {len(report['issues'])}")

        if report["class_distribution"]:
            print(f"   • Distribution des classes: {report['class_distribution']}")

        return report


def parse_arguments():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="🔄 Conversion d'annotations vers le format YOLO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  # COCO vers YOLO
  python scripts/convert_to_yolo.py --format coco --input annotations.json --images images/ --output labels/
  
  # Pascal VOC vers YOLO
  python scripts/convert_to_yolo.py --format voc --input xml_dir/ --images images/ --output labels/
  
  # CSV vers YOLO
  python scripts/convert_to_yolo.py --format csv --input annotations.csv --images images/ --output labels/
  
  # Validation des annotations YOLO
  python scripts/convert_to_yolo.py --validate labels/ --images images/
        """,
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["coco", "voc", "csv"],
        help="Format source des annotations",
    )
    parser.add_argument(
        "--input", type=str, help="Fichier ou répertoire source des annotations"
    )
    parser.add_argument("--images", type=str, help="Répertoire contenant les images")
    parser.add_argument(
        "--output", type=str, help="Répertoire de sortie pour les annotations YOLO"
    )
    parser.add_argument(
        "--validate",
        type=str,
        help="Valider les annotations YOLO dans le répertoire donné",
    )
    parser.add_argument(
        "--classes",
        type=str,
        help='Fichier JSON avec le mapping des classes (format: {"nom_classe": id})',
    )
    parser.add_argument(
        "--csv-columns", type=str, help="Fichier JSON avec le mapping des colonnes CSV"
    )

    return parser.parse_args()


def load_class_mapping(classes_file: str) -> Dict[str, int]:
    """Charge le mapping des classes depuis un fichier JSON."""
    try:
        with open(classes_file, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Erreur chargement classes {classes_file}: {e}")
        return {"fracture": 0}


def main():
    """Fonction principale."""
    args = parse_arguments()

    print("🔄 Convertisseur d'annotations vers YOLO")

    # Charger le mapping des classes
    class_mapping = {"fracture": 0}  # Défaut
    if args.classes:
        class_mapping = load_class_mapping(args.classes)

    # Mode validation uniquement
    if args.validate:
        converter = AnnotationConverter(class_mapping)
        report = converter.validate_yolo_annotations(args.validate, args.images)

        if report["issues"]:
            print(f"\n⚠️  Problèmes détectés:")
            for issue in report["issues"][:10]:  # Afficher les 10 premiers
                print(f"   • {issue}")
            if len(report["issues"]) > 10:
                print(f"   ... et {len(report['issues']) - 10} autres problèmes")
        else:
            print(f"✅ Validation réussie: toutes les annotations sont valides")

        return

    # Vérifier les arguments requis
    if not all([args.format, args.input, args.images, args.output]):
        print("❌ Arguments manquants. Utilisez --help pour voir l'aide.")
        return

    # Initialiser le convertisseur
    converter = AnnotationConverter(class_mapping)

    # Effectuer la conversion selon le format
    success = False

    if args.format == "coco":
        success = converter.coco_to_yolo(args.input, args.images, args.output)

    elif args.format == "voc":
        success = converter.pascal_voc_to_yolo(args.input, args.images, args.output)

    elif args.format == "csv":
        # Charger le mapping des colonnes CSV si fourni
        csv_columns = None
        if args.csv_columns:
            try:
                with open(args.csv_columns, "r") as f:
                    csv_columns = json.load(f)
            except Exception as e:
                print(f"⚠️  Erreur chargement colonnes CSV: {e}")

        success = converter.csv_to_yolo(
            args.input, args.images, args.output, csv_columns
        )

    if success:
        print(f"🎉 Conversion terminée avec succès!")
        print(f"📁 Annotations YOLO disponibles dans: {args.output}")

        # Validation automatique
        print(f"\n🔍 Validation des annotations créées...")
        report = converter.validate_yolo_annotations(args.output, args.images)

        if not report["issues"]:
            print(f"✅ Toutes les annotations sont valides!")
        else:
            print(f"⚠️  {len(report['issues'])} problème(s) détecté(s)")
    else:
        print(f"❌ Échec de la conversion")


if __name__ == "__main__":
    main()
