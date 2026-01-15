import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import random
import os
import json

# Import your architecture
from model import EquivariantTankHunter

# --- SETTINGS ---
MODEL_PATH = 'tank_hunter_model.pth'
DATA_FOLDER = 'Data/image_patches'
LABEL_FILE = 'Data/labels.json'

# --- 1. LOAD THE BRAIN ---
print(f"🧠 Loading the trained model from {MODEL_PATH}...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the architecture
model = EquivariantTankHunter().to(device)

# --- FIX IS HERE: strict=False ---
# Esto le dice a la IA: "Si falta el buffer temporal 'filter', no entres en pánico, recálculalo."
model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)

model.eval() # Switch to "Exam Mode"
print("✅ Model loaded successfully! (Calculated filters generated on the fly)")

# --- 2. PREPARE THE DATA ---
with open(LABEL_FILE, 'r') as f:
    labels_data = json.load(f)

# Separate tanks and empty fields
tank_images = [item['file_name'] for item in labels_data if 'Tank' in str(item['label'])]
empty_images = [item['file_name'] for item in labels_data if 'Skip' in str(item['label'])]

# Define the image transformer (Must be 64x64)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

def predict_image(filename):
    path = os.path.join(DATA_FOLDER, filename)
    
    try:
        image = Image.open(path).convert('RGB')
    except:
        return None, 0

    # Preprocess
    img_tensor = transform(image).unsqueeze(0).to(device) 
    
    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        tank_prob = probabilities[0][1].item() 
        
    return image, tank_prob

# --- 3. RUN THE TEST ---
print("\n🔎 Running random tests...")

# Pick 3 Tanks and 3 Empty Fields
# (Added a safety check in case the list is empty, though it shouldn't be)
if len(tank_images) < 3 or len(empty_images) < 3:
    print("⚠️ Warning: Not enough images found to run a full test.")
    test_files = tank_images + empty_images
else:
    test_files = random.sample(tank_images, 3) + random.sample(empty_images, 3)

random.shuffle(test_files)

plt.figure(figsize=(12, 8))

valid_count = 0
for i, filename in enumerate(test_files):
    img, probability = predict_image(filename)
    
    if img is None:
        continue

    # Visualization Logic
    prediction = "TANK" if probability > 0.5 else "EMPTY"
    confidence = probability if probability > 0.5 else 1 - probability
    
    is_actually_tank = filename in tank_images
    actual_label = "TANK" if is_actually_tank else "EMPTY"
    
    color = 'green' if prediction == actual_label else 'red'
    
    plt.subplot(2, 3, valid_count+1)
    plt.imshow(img)
    plt.title(f"AI Says: {prediction} ({confidence:.0%})\nActual: {actual_label}", color=color, fontweight='bold')
    plt.axis('off')
    valid_count += 1
    if valid_count >= 6: break

plt.tight_layout()
plt.show()

print("📸 Check the popup window!")