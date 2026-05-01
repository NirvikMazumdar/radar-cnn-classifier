from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


# =========================
# PATHS
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "radar_cnn_model.pth"


# =========================
# MODEL (same as train.py)
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
        return self.classifier(x)


# =========================
# LOAD MODEL
# =========================

def load_model():
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

    return model, class_names, image_size, device


# =========================
# PREDICT SINGLE IMAGE
# =========================

def predict_image(image_path):
    model, class_names, image_size, device = load_model()

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_index = torch.max(probabilities, 1)

    predicted_label = class_names[predicted_index.item()]
    confidence_score = confidence.item()

    print("\n🔍 Prediction Result")
    print(f"Image: {image_path}")
    print(f"Predicted Class: {predicted_label}")
    print(f"Confidence: {confidence_score:.4f}")


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    # 🔹 CHANGE THIS PATH TO ANY IMAGE YOU WANT
    test_image_path = input("Enter image path: ").strip()

    if not Path(test_image_path).exists():
        print("❌ Invalid path")
    else:
        predict_image(test_image_path)