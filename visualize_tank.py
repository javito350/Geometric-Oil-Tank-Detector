import os
import json
import random
from PIL import Image
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
base_folder = 'Data'
image_folder = os.path.join(base_folder, 'image_patches')
labels_file = os.path.join(base_folder, 'labels.json')

# --- LOGIC ---
try:
    print("📖 Reading label file...")
    with open(labels_file, 'r') as f:
        data = json.load(f) # It's a LIST

    # Filter: Find items where the label string contains "Tank"
    # The label looks like: {'Tank': ...} or 'Floating Head Tank'
    real_tanks = []
    for item in data:
        label_content = str(item['label']) # Convert dictionary/string to text
        if 'Tank' in label_content or 'tank' in label_content:
            real_tanks.append(item['file_name'])

    print(f"✅ Found {len(real_tanks)} images with Tanks inside.")

    if len(real_tanks) > 0:
        # Pick a random one
        image_name = random.choice(real_tanks)
        full_path = os.path.join(image_folder, image_name)
        
        # Show it
        print(f"📸 Opening: {image_name}")
        img = Image.open(full_path)
        
        plt.figure(figsize=(5,5))
        plt.imshow(img)
        plt.title(f"Target Detected!\nFile: {image_name}")
        plt.axis('off')
        plt.show()
    else:
        print("⚠️ No tanks found. Check the labels logic.")

except FileNotFoundError:
    print("❌ Error: Could not find 'labels.json' or 'Data' folder.")