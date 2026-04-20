from flask import Flask, request, jsonify
import torch
import timm
from PIL import Image
import torchvision.transforms as transforms
from config import CLASSES
import os

app = Flask(__name__)

# ✅ 修复路径（Render必须这样写）
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "fracture_model.pt")

# ✅ 初始化模型
model = timm.create_model(
    "mobilenetv3_small_100",
    pretrained=False,
    num_classes=len(CLASSES)
)

# ✅ 关键修复（PyTorch 2.6+ 必须）
model.load_state_dict(
    torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
)

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

# ✅ Render必须监听这个端口
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
