from pathlib import Path
import random
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from flask import Flask, render_template_string, send_from_directory

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "radar_cnn_model.pth"
TEST_DIR = PROJECT_ROOT / "data" / "processed" / "test"

app = Flask(__name__)

class RadarCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(MODEL_PATH, map_location=device)

class_names = checkpoint["class_names"]
image_size = checkpoint["image_size"]

model = RadarCNN(num_classes=len(class_names)).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred_idx = torch.max(probs, 1)

    return class_names[pred_idx.item()], confidence.item()


@app.route("/")
def index():
    image_paths = list(TEST_DIR.glob("*/*.png"))
    image_path = random.choice(image_paths)

    true_label = image_path.parent.name
    predicted_label, confidence = predict_image(image_path)
    correct = true_label == predicted_label

    relative_img_path = image_path.relative_to(PROJECT_ROOT).as_posix()

    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Radar CNN Demo</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background: #f4f6f8;
                text-align: center;
                padding: 40px;
            }
            .card {
                background: white;
                padding: 30px;
                border-radius: 16px;
                max-width: 700px;
                margin: auto;
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            }
            img {
                width: 300px;
                border-radius: 10px;
                margin: 20px;
            }
            button {
                background: #0066ff;
                color: white;
                border: none;
                padding: 14px 24px;
                border-radius: 8px;
                font-size: 16px;
                cursor: pointer;
            }
            .correct {
                color: green;
                font-size: 24px;
                font-weight: bold;
            }
            .wrong {
                color: red;
                font-size: 24px;
                font-weight: bold;
            }
            .info {
                font-size: 18px;
                margin-top: 10px;
            }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>Radar CNN Classification Demo</h1>

            <img src="/file/{{ img_path }}">

            <div class="info">True Label: <b>{{ true_label }}</b></div>
            <div class="info">Predicted Label: <b>{{ predicted_label }}</b></div>
            <div class="info">Confidence: <b>{{ confidence }}%</b></div>

            {% if correct %}
                <p class="correct">✅ Correct Classification</p>
            {% else %}
                <p class="wrong">❌ Wrong Classification</p>
            {% endif %}

            <form action="/" method="get">
                <button type="submit">Randomize</button>
            </form>
        </div>
    </body>
    </html>
    """

    return render_template_string(
        html,
        img_path=relative_img_path,
        true_label=true_label,
        predicted_label=predicted_label,
        confidence=round(confidence * 100, 2),
        correct=correct
    )


@app.route("/file/<path:filename>")
def serve_file(filename):
    return send_from_directory(PROJECT_ROOT, filename)


if __name__ == "__main__":
    app.run(debug=True)