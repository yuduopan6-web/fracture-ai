from flask import Flask, request, jsonify
import torch
from PIL import Image
import torchvision.transforms as transforms
from config import CLASSES
import os

app = Flask(__name__)

# ✅ 正确路径
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "fracture_model.pt")

# ✅ 直接加载 TorchScript 模型（关键！）
model = torch.jit.load(MODEL_PATH, map_location="cpu")
model.eval()

# ✅ 图片预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.route("/")
def home():
    return "Fracture AI is running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["file"]
        img = Image.open(file).convert("RGB")
        img = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(img)
            pred = torch.argmax(output, 1).item()

        return jsonify({
            "result": CLASSES[pred],
            "status": "success"
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "failed"
        })

# ✅ Render端口适配
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
