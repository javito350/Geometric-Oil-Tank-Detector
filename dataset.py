import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import json
import torchvision.transforms as transforms

class TankDataset(Dataset):
    def __init__(self, root_dir='Data', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # 1. Load the JSON (It is a LIST)
        json_path = os.path.join(root_dir, 'labels.json')
        with open(json_path, 'r') as f:
            self.labels_data = json.load(f)
            
        self.image_folder = os.path.join(root_dir, 'image_patches')
        print(f"📂 Dataset loaded: {len(self.labels_data)} items found.")

        # Optional: Print unique labels just so we know what we are dealing with
        try:
            unique_labels = set(item['label'] for item in self.labels_data)
            print(f"🧐 Labels found in file: {unique_labels}")
        except:
            pass

    def __len__(self):
        return len(self.labels_data)

    def __getitem__(self, idx):
        # 1. Get the item from the LIST using the index
        item = self.labels_data[idx]
        
        # Extract info using the keys we saw in check.py
        img_name = item['file_name'] 
        label_str = item['label']    
        
        # 2. Smart Labeling
        # If the label says "Tank" (e.g., "Floating Head Tank"), it's a 1.
        # If it says "Skip" or "Background", it's a 0.
        if 'Tank' in label_str or 'tank' in label_str:
            label = 1
        else:
            label = 0 

        # 3. Open Image
        img_path = os.path.join(self.image_folder, img_name)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            # If an image is broken, return a black square so training doesn't crash
            image = Image.new('RGB', (256, 256))

        # 4. Transform to Tensor
        if self.transform:
            image = self.transform(image)
        
        return image, label

# --- TEST BLOCK ---
if __name__ == '__main__':
    # Define basic transform
    my_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # Test the Dataset
    dataset = TankDataset(transform=my_transforms)
    
    # Check the first item
    print("\n--- TEST RESULT ---")
    image, label = dataset[0]
    print(f"✅ It works! Image shape: {image.shape}")
    print(f"🏷️ Label: {label}")
    
    # Check a few more to see if we find a tank
    print("Checking first 10 items for a Tank...")
    for i in range(10):
        _, lbl = dataset[i]
        print(f"Item {i}: Label {lbl} ({dataset.labels_data[i]['label']})")