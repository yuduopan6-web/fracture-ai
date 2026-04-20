from flask import Flask, request, jsonify
import torch
import timm
from PIL import Image
import torchvision.transforms as transforms
from config import CLASSES

app = Flask(__name__)

MODEL_PATH = "../models/fracture_model.pt"

model = timm.create_model("mobilenetv3_small_100", pretrained=False, num_classes=len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    img = Image.open(file).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        pred = torch.argmax(output, 1).item()

    return jsonify({"result": CLASSES[pred]})

if __name__ == "__main__":
    import os
port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)
