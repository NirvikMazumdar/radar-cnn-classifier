from pathlib import Path
import tarfile
import random
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from sklearn.model_selection import train_test_split


# =========================
# PATHS
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
VISUALIZATION_DIR = PROJECT_ROOT / "outputs" / "preprocessing_examples"

TAR_PATH = RAW_DIR / "VOCtrainval_11-May-2012.tar"
VOC_ROOT = RAW_DIR / "VOCdevkit" / "VOC2012"

JPEG_DIR = VOC_ROOT / "JPEGImages"
ANNOTATION_DIR = VOC_ROOT / "Annotations"


# =========================
# SETTINGS
# =========================

IMAGE_SIZE = 128
RANDOM_SEED = 42
MAX_VISUAL_EXAMPLES_PER_SPLIT = 50

CLASS_MAP = {
    "person": "human",

    "car": "vehicle",
    "bus": "vehicle",
    "train": "vehicle",
    "motorbike": "vehicle",
    "bicycle": "vehicle",

    "aeroplane": "vehicle",
    "boat": "vehicle",
}

FINAL_CLASSES = ["human", "vehicle", "other"]


# =========================
# FUNCTIONS
# =========================

def extract_tar():
    if VOC_ROOT.exists():
        print("✅ VOC dataset already extracted.")
        return

    if not TAR_PATH.exists():
        raise FileNotFoundError(f"❌ TAR file not found: {TAR_PATH}")

    print(f"📦 Extracting: {TAR_PATH}")

    with tarfile.open(TAR_PATH, "r") as tar:
        tar.extractall(path=RAW_DIR)

    print("✅ Extraction complete.")


def read_voc_annotation(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    objects = []

    for obj in root.findall("object"):
        name = obj.find("name").text.lower().strip()
        bbox = obj.find("bndbox")

        xmin = int(float(bbox.find("xmin").text))
        ymin = int(float(bbox.find("ymin").text))
        xmax = int(float(bbox.find("xmax").text))
        ymax = int(float(bbox.find("ymax").text))

        objects.append({
            "name": name,
            "bbox": [xmin, ymin, xmax, ymax]
        })

    return objects


def decide_image_label(objects):
    """
    Dominant-object labeling:
    The image label is decided by the class with the largest total bounding-box area.
    This avoids labeling a large vehicle image as human just because a small person appears.
    """

    class_area = {
        "human": 0,
        "vehicle": 0,
        "other": 0
    }

    for obj in objects:
        xmin, ymin, xmax, ymax = obj["bbox"]
        area = max(0, xmax - xmin) * max(0, ymax - ymin)

        mapped_class = CLASS_MAP.get(obj["name"], "other")
        class_area[mapped_class] += area

    return max(class_area, key=class_area.get)


def create_synthetic_radar_image(image_path, objects):
    """
    Class-conditioned synthetic radar-like image.

    This is NOT real radar data.
    It creates radar-inspired signatures:
    - human: weaker blob + vertical micro-Doppler-like streaks
    - vehicle: stronger, wider, horizontal reflection
    - other: weak scattered/static reflections
    """

    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    radar = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    for obj in objects:
        xmin, ymin, xmax, ymax = obj["bbox"]
        mapped_class = CLASS_MAP.get(obj["name"], "other")

        cx = int(((xmin + xmax) / 2) / width * IMAGE_SIZE)
        cy = int(((ymin + ymax) / 2) / height * IMAGE_SIZE)

        box_w = max(3, int((xmax - xmin) / width * IMAGE_SIZE))
        box_h = max(3, int((ymax - ymin) / height * IMAGE_SIZE))

        if mapped_class == "human":
            blob_w = max(4, int(box_w * 0.45))
            blob_h = max(8, int(box_h * 0.70))
            intensity = random.uniform(0.35, 0.65)

            cv2.ellipse(
                radar,
                center=(cx, cy),
                axes=(blob_w // 2, blob_h // 2),
                angle=random.randint(-15, 15),
                startAngle=0,
                endAngle=360,
                color=intensity,
                thickness=-1
            )

            for _ in range(random.randint(3, 7)):
                x_offset = random.randint(-blob_w, blob_w)
                x = np.clip(cx + x_offset, 0, IMAGE_SIZE - 1)

                y1 = np.clip(cy - blob_h, 0, IMAGE_SIZE - 1)
                y2 = np.clip(cy + blob_h, 0, IMAGE_SIZE - 1)

                cv2.line(
                    radar,
                    (int(x), int(y1)),
                    (int(x), int(y2)),
                    color=random.uniform(0.20, 0.45),
                    thickness=1
                )

        elif mapped_class == "vehicle":
            blob_w = max(12, int(box_w * 0.90))
            blob_h = max(5, int(box_h * 0.35))
            intensity = random.uniform(0.75, 1.0)

            cv2.ellipse(
                radar,
                center=(cx, cy),
                axes=(blob_w // 2, blob_h // 2),
                angle=random.randint(-8, 8),
                startAngle=0,
                endAngle=360,
                color=intensity,
                thickness=-1
            )

            for _ in range(random.randint(2, 4)):
                y_offset = random.randint(-blob_h, blob_h)
                y = np.clip(cy + y_offset, 0, IMAGE_SIZE - 1)

                x1 = np.clip(cx - blob_w, 0, IMAGE_SIZE - 1)
                x2 = np.clip(cx + blob_w, 0, IMAGE_SIZE - 1)

                cv2.line(
                    radar,
                    (int(x1), int(y)),
                    (int(x2), int(y)),
                    color=random.uniform(0.45, 0.80),
                    thickness=1
                )

        else:
            for _ in range(random.randint(2, 6)):
                sx = np.clip(cx + random.randint(-box_w, box_w), 0, IMAGE_SIZE - 1)
                sy = np.clip(cy + random.randint(-box_h, box_h), 0, IMAGE_SIZE - 1)

                cv2.circle(
                    radar,
                    center=(int(sx), int(sy)),
                    radius=random.randint(1, 3),
                    color=random.uniform(0.15, 0.40),
                    thickness=-1
                )

    radar = cv2.GaussianBlur(radar, (9, 9), 0)

    noise = np.random.normal(0, 0.035, radar.shape).astype(np.float32)
    radar = radar + noise

    radar = np.clip(radar, 0, 1)
    radar = (radar * 255).astype(np.uint8)

    radar_colored = cv2.applyColorMap(radar, cv2.COLORMAP_JET)
    radar_colored = cv2.cvtColor(radar_colored, cv2.COLOR_BGR2RGB)

    return Image.fromarray(radar_colored)


def create_side_by_side_visual(image_path, objects, radar_img, label, save_path):
    original_full = Image.open(image_path).convert("RGB")
    original_width, original_height = original_full.size

    original = original_full.resize((IMAGE_SIZE, IMAGE_SIZE))
    draw = ImageDraw.Draw(original)

    for obj in objects:
        xmin, ymin, xmax, ymax = obj["bbox"]
        obj_name = obj["name"]

        xmin = int(xmin / original_width * IMAGE_SIZE)
        xmax = int(xmax / original_width * IMAGE_SIZE)
        ymin = int(ymin / original_height * IMAGE_SIZE)
        ymax = int(ymax / original_height * IMAGE_SIZE)

        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
        draw.text((xmin, max(0, ymin - 12)), obj_name, fill="red")

    radar_img = radar_img.resize((IMAGE_SIZE, IMAGE_SIZE))

    header_height = 28
    combined_width = IMAGE_SIZE * 2
    combined_height = IMAGE_SIZE + header_height

    combined = Image.new("RGB", (combined_width, combined_height), "white")
    combined.paste(original, (0, header_height))
    combined.paste(radar_img, (IMAGE_SIZE, header_height))

    draw_combined = ImageDraw.Draw(combined)
    draw_combined.text(
        (5, 7),
        f"Original + boxes | Synthetic radar-like | Label: {label}",
        fill="black"
    )

    combined.save(save_path)


def clear_processed_dir():
    if PROCESSED_DIR.exists():
        print("🧹 Existing processed data found. Keeping folder but overwriting images.")

    for split in ["train", "val", "test"]:
        for cls in FINAL_CLASSES:
            folder = PROCESSED_DIR / split / cls
            folder.mkdir(parents=True, exist_ok=True)

            for file in folder.glob("*"):
                file.unlink()


def clear_visualization_dir():
    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

    for file in VISUALIZATION_DIR.glob("*"):
        file.unlink()


def preprocess_dataset():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    extract_tar()
    clear_processed_dir()
    clear_visualization_dir()

    xml_files = list(ANNOTATION_DIR.glob("*.xml"))

    if not xml_files:
        raise FileNotFoundError(f"❌ No XML files found in: {ANNOTATION_DIR}")

    dataset_items = []

    print("🔍 Reading VOC annotations...")

    for xml_path in tqdm(xml_files):
        image_name = xml_path.stem + ".jpg"
        image_path = JPEG_DIR / image_name

        if not image_path.exists():
            continue

        objects = read_voc_annotation(xml_path)

        if not objects:
            continue

        label = decide_image_label(objects)

        dataset_items.append({
            "image_path": image_path,
            "objects": objects,
            "label": label,
            "name": xml_path.stem
        })

    print(f"✅ Total usable images: {len(dataset_items)}")

    train_items, temp_items = train_test_split(
        dataset_items,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=[item["label"] for item in dataset_items]
    )

    val_items, test_items = train_test_split(
        temp_items,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=[item["label"] for item in temp_items]
    )

    splits = {
        "train": train_items,
        "val": val_items,
        "test": test_items
    }

    visual_counter = {
        "train": 0,
        "val": 0,
        "test": 0
    }

    print("🛰️ Creating class-conditioned synthetic radar-like images...")

    for split_name, items in splits.items():
        for item in tqdm(items, desc=f"Processing {split_name}"):
            radar_img = create_synthetic_radar_image(
                item["image_path"],
                item["objects"]
            )

            save_dir = PROCESSED_DIR / split_name / item["label"]
            save_path = save_dir / f"{item['name']}.png"
            radar_img.save(save_path)

            if visual_counter[split_name] < MAX_VISUAL_EXAMPLES_PER_SPLIT:
                visual_save_path = VISUALIZATION_DIR / f"{split_name}_{item['label']}_{item['name']}_comparison.png"

                create_side_by_side_visual(
                    image_path=item["image_path"],
                    objects=item["objects"],
                    radar_img=radar_img,
                    label=item["label"],
                    save_path=visual_save_path
                )

                visual_counter[split_name] += 1

    print("✅ Preprocessing complete.")
    print(f"📁 Saved processed dataset to: {PROCESSED_DIR}")
    print(f"🖼️ Saved comparison examples to: {VISUALIZATION_DIR}")


if __name__ == "__main__":
    preprocess_dataset()