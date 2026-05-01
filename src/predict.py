from pathlib import Path
import random

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw
from tqdm import tqdm


# =========================
# PATHS
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_DIR = PROJECT_ROOT / "data" / "processed" / "test"
MODEL_PATH = PROJECT_ROOT / "models" / "radar_cnn_model.pth"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "predictions"


# =========================
# MODEL
# =========================

class RadarCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(RadarCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# =========================
# PREDICT FUNCTION
# =========================

def predict_test_images():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    class_names = checkpoint["class_names"]
    image_size = checkpoint["image_size"]

    model = RadarCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    image_paths = list(TEST_DIR.glob("*/*.png"))

    if not image_paths:
        raise FileNotFoundError(f"❌ No test images found in: {TEST_DIR}")

    random.shuffle(image_paths)
    image_paths = image_paths[:30]

    print(f"🔍 Predicting {len(image_paths)} test images...")

    correct = 0
    total = 0

    for image_path in tqdm(image_paths):
        true_label = image_path.parent.name

        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_index = torch.max(probabilities, 1)

        predicted_label = class_names[predicted_index.item()]
        confidence_score = confidence.item()

        if predicted_label == true_label:
            correct += 1

        total += 1

        output_image = image.copy()
        draw = ImageDraw.Draw(output_image)

        text = f"True: {true_label} | Pred: {predicted_label} | Conf: {confidence_score:.2f}"

        draw.rectangle([0, 0, output_image.width, 22], fill=(0, 0, 0))
        draw.text((5, 5), text, fill=(255, 255, 255))

        save_name = f"{image_path.stem}_pred_{predicted_label}.png"
        save_path = OUTPUT_DIR / save_name

        output_image.save(save_path)

    accuracy = 100 * correct / total

    print(f"✅ Prediction complete.")
    print(f"📊 Sample Test Accuracy: {accuracy:.2f}%")
    print(f"📁 Prediction images saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    predict_test_images()