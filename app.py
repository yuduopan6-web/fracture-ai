from flask import Flask, request, jsonify
import torch
import timm
from PIL import Image
import torchvision.transforms as transforms
from config import CLASSES
import os

app = Flask(__name__)

# ✅ 关键：绝对路径（解决 Render 找不到模型）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fracture_model.pt")

# ✅ 加载模型
model = timm.create_model(
    "mobilenetv3_small_100",
    pretrained=False,
    num_classes=len(CLASSES)
)

model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# ✅ 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ✅ 接口
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        img = Image.open(file).convert("RGB")
        img = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(img)
            pred = torch.argmax(output, 1).item()

        return jsonify({"result": CLASSES[pred]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ✅ 关键：适配 Render 端口
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
