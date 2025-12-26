import os
import cv2
import torch
import numpy as np
from PIL import Image
from model.u2net import U2NET

INPUT_DIR = "images"
OUTPUT_DIR = "mask"
MODEL_PATH = "weights/u2net_human_seg.pth"
IMAGE_SIZE = 320
THRESHOLD = 0.80  # Increase for stricter segmentation

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load U2NET
model = U2NET(3, 1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

def preprocess(img_path):
    img = Image.open(img_path).convert("RGB")
    original_size = img.size
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img = np.array(img) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    return img.to(device), original_size

def postprocess(pred, original_size):
    pred = pred[:, 0, :, :]
    pred = (pred - pred.min()) / (pred.max() - pred.min())
    mask = (pred > THRESHOLD).float() * 255
    mask = mask.cpu().numpy().astype(np.uint8)[0]
    mask = cv2.resize(mask, original_size)
    return mask

# Process folder
for file in os.listdir(INPUT_DIR):
    if not file.lower().endswith((".jpg", ".png", ".jpeg", ".bmp", ".webp")):
        continue

    img_path = os.path.join(INPUT_DIR, file)
    out_path = os.path.join(OUTPUT_DIR, file)

    inp, original_size = preprocess(img_path)

    with torch.no_grad():
        d1, *_ = model(inp)

    mask = postprocess(d1, original_size)
    cv2.imwrite(out_path, mask)

    print(f"✅ Saved mask: {out_path}")

mask_images = os.listdir(OUTPUT_DIR)
print(f"\nTotal Mask Images: {len(mask_images)}")