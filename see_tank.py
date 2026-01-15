import os
from PIL import Image
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
# We look for the folder where you unzipped the data.
possible_paths = [
    os.path.join('Data', 'image_patches'),  # <--- Added this for you!
    os.path.join('Data', 'train'),
    os.path.join('Data', 'train_images'),
    os.path.join('Data', 'Oil Tanks', 'image_patches') # Just in case
]

target_folder = None
for path in possible_paths:
    if os.path.exists(path):
        target_folder = path
        break

# --- THE LOGIC ---
if target_folder:
    print(f"✅ FOUND IT! Reading images from: {target_folder}")
    
    # Get all files
    files = [f for f in os.listdir(target_folder) if f.endswith('.jpg') or f.endswith('.png')]
    
    if len(files) > 0:
        # Pick the first one
        first_image = files[0]
        full_path = os.path.join(target_folder, first_image)

        # Show it
        print(f"📸 Opening: {first_image}")
        img = Image.open(full_path)
        plt.imshow(img)
        plt.title(f"My First Tank: {first_image}")
        plt.axis('off')
        plt.show()
    else:
        print(f"⚠️ The folder '{target_folder}' exists, but there are no images inside!")
else:
    print("❌ ERROR: Could not find the image folder.")
    print("Did you copy 'image_patches' into your 'Data' folder?")