from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from PIL import Image
import torchvision.transforms as transforms
from config import CLASSES

app = Flask(__name__)
CORS(app)  # ✅ 解决跨域问题（关键）

# ✅ 正确路径（Render 上必须这样写）
MODEL_PATH = "models/fracture_model.pt"

# ✅ 加载 TorchScript 模型（你现在用的是这个格式）
model = torch.jit.load(MODEL_PATH, map_location="cpu")
model.eval()

# ✅ 图片预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ✅ 测试接口（防止打开首页报错）
@app.route("/")
def home():
    return "API is running"

# ✅ 预测接口
@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["file"]
        img = Image.open(file).convert("RGB")
        img = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(img)
            pred = torch.argmax(output, 1).item()

        return jsonify({"result": CLASSES[pred]})

    except Exception as e:
        return jsonify({"error": str(e)})

# ✅ Render 必须用这个端口启动
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
