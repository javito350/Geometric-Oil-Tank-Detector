import torch
from e2cnn import gspaces
from e2cnn import nn as enn

class EquivariantTankHunter(torch.nn.Module):
    def __init__(self):
        super(EquivariantTankHunter, self).__init__()
        
        # --- THE ALGEBRA PART (Group Theory) ---
        # We define the symmetry group C8 (Rotations by 45 degrees)
        self.r2_act = gspaces.Rot2dOnR2(N=8)
        
        # --- INPUT LAYER ---
        in_type = enn.FieldType(self.r2_act, 3 * [self.r2_act.trivial_repr])
        
        # --- HIDDEN LAYERS ---
        # We use "Regular Representation" (regular_repr)
        self.feat_type = enn.FieldType(self.r2_act, 24 * [self.r2_act.regular_repr])
        
        # Layer 1
        self.block1 = enn.SequentialModule(
            enn.R2Conv(in_type, self.feat_type, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(self.feat_type),
            enn.ReLU(self.feat_type, inplace=True)
        )
        
        # Layer 2
        self.block2 = enn.SequentialModule(
            enn.R2Conv(self.feat_type, self.feat_type, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(self.feat_type),
            enn.ReLU(self.feat_type, inplace=True)
        )
        
        # Layer 3: Geometric Max Pooling
        # FIXED HERE: Added kernel_size=2 (Shrinks image from 64x64 -> 32x32)
        self.pool = enn.PointwiseMaxPool(self.feat_type, kernel_size=2)
        
        # --- OUTPUT LAYER (Classification) ---
        # Group Pooling: Collapses the 8 rotations into 1 invariant feature
        self.gpool = enn.GroupPooling(self.feat_type)
        
        # Linear layer: Input is 24 channels (from feat_type), Output is 2 classes
        self.linear = torch.nn.Linear(24, 2) 

    def forward(self, x):
        # 1. Wrap image into Geometric Tensor
        x = enn.GeometricTensor(x, enn.FieldType(self.r2_act, 3 * [self.r2_act.trivial_repr]))
        
        # 2. Equivariant Layers
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        
        # 3. Collapse geometry (Group Pooling)
        x = self.gpool(x)
        
        # 4. Unwrap and Classify
        x = x.tensor 
        
        # Global Average Pooling (squash remaining pixels to 1x1)
        x = x.mean(dim=[2, 3]) 
        
        x = self.linear(x)
        return x

# --- VERIFICATION BLOCK ---
if __name__ == '__main__':
    print("🏗️ Building the Group Equivariant Model...")
    model = EquivariantTankHunter()
    
    # Fake image: Batch of 4 images, 3 channels (RGB), 64x64 pixels
    fake_image = torch.randn(4, 3, 64, 64)
    
    print("🔄 Passing a fake image through the layers...")
    output = model(fake_image)
    
    print(f"✅ Success! Output shape: {output.shape}")
    print("Expected: torch.Size([4, 2])")