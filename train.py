import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import time

# Import your custom files
from dataset import TankDataset
from model import EquivariantTankHunter

# --- 1. SETTINGS (Hyperparameters) ---
BATCH_SIZE = 32      # How many images to learn from at once
LEARNING_RATE = 0.001 # How fast to change its mind
EPOCHS = 5           # How many times to read the entire dataset
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"⚙️ Computation Device: {DEVICE}")

# --- 2. PREPARE DATA ---
print("📦 Loading Data...")
# We resize to 64x64 to make training fast on your laptop
my_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

full_dataset = TankDataset(root_dir='Data', transform=my_transforms)

# Split: 80% for Training (Studying), 20% for Validation (Exam)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_data, val_data = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

print(f"✅ Ready to train on {len(train_data)} images. Validation on {len(val_data)} images.")

# --- 3. INITIALIZE MODEL ---
model = EquivariantTankHunter().to(DEVICE)

# OPTIMIZER: The algorithm that updates the math (Adam is the standard best)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# LOSS FUNCTION: How we measure mistakes.
# "weight=torch.tensor([1.0, 50.0])" means:
# "Pay 50x more attention if you miss a TANK than if you miss an empty field."
weights = torch.tensor([1.0, 30.0]).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=weights)

# --- 4. TRAINING LOOP ---
print("\n🚀 STARTING TRAINING ENGINE...")
start_time = time.time()

for epoch in range(EPOCHS):
    model.train() # Switch to "Study Mode"
    running_loss = 0.0
    
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        # A. Zero the gradients (reset previous calculations)
        optimizer.zero_grad()
        
        # B. Forward Pass (Guess)
        outputs = model(images)
        
        # C. Calculate Error (Loss)
        loss = criterion(outputs, labels)
        
        # D. Backward Pass (Learn)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Print update every 50 batches
        if (i+1) % 50 == 0:
            print(f"   [Epoch {epoch+1}, Batch {i+1}] Loss: {loss.item():.4f}")

    # End of Epoch Stats
    avg_loss = running_loss / len(train_loader)
    print(f"🏁 Epoch {epoch+1}/{EPOCHS} Finished. Average Loss: {avg_loss:.4f}")

# --- 5. SAVE THE BRAIN ---
print("\n💾 Saving the trained model...")
torch.save(model.state_dict(), 'tank_hunter_model.pth')
print("✅ Model saved as 'tank_hunter_model.pth'")

total_time = (time.time() - start_time) / 60
print(f"⏱️ Total training time: {total_time:.2f} minutes.")