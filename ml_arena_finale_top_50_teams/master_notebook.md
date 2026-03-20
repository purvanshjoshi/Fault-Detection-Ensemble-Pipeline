# [IEEE TEMPLATE: DROP-IN MASTER CODE]

Use these cells to replace the placeholder sections in your template notebook.

---

### Replace Section: "Implementation & Training"

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.transforms import v2
from PIL import Image
from tqdm.auto import tqdm
import numpy as np

# --- Noble Configuration ---
class Config:
    IMAGE_PATH = image_dir # Uses variable from your template
    NUM_CLASSES = 397
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-4
    EPOCHS = 15
    IMAGE_SIZE = 224
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Advanced Dataset & Augmentations ---
def get_transforms(phase='train'):
    if phase == 'train':
        return v2.Compose([
            v2.RandomResizedCrop(size=(Config.IMAGE_SIZE, Config.IMAGE_SIZE), antialias=True),
            v2.RandomHorizontalFlip(),
            v2.RandAugment(num_ops=2, magnitude=12),
            v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return v2.Compose([
        v2.Resize(size=(Config.IMAGE_SIZE, Config.IMAGE_SIZE), antialias=True),
        v2.CenterCrop(size=(Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

class MLDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, is_test=False):
        self.df, self.image_dir = df, image_dir
        self.transform, self.is_test = transform, is_test
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['IMAGE'])
        try: image = Image.open(img_path).convert("RGB")
        except: image = Image.new("RGB", (Config.IMAGE_SIZE, Config.IMAGE_SIZE))
        if self.transform: image = self.transform(image)
        if self.is_test: return image, row['IMAGE']
        return image, int(row['LABEL'])

# Mixup/Cutmix Logic
cutmix = v2.CutMix(num_classes=Config.NUM_CLASSES)
mixup = v2.MixUp(num_classes=Config.NUM_CLASSES)
mix_op = v2.RandomChoice([cutmix, mixup])

# --- Model & Training Engine ---
def build_model():
    model = models.convnext_tiny(weights="IMAGENET1K_V1")
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, Config.NUM_CLASSES)
    return model.to(Config.DEVICE)

# Training Loop
val_df_split = train_df.sample(frac=0.2, random_state=42)
train_df_split = train_df.drop(val_df_split.index)

train_loader = DataLoader(MLDataset(train_df_split, Config.IMAGE_PATH, get_transforms('train')), batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(MLDataset(val_df_split, Config.IMAGE_PATH, get_transforms('val')), batch_size=Config.BATCH_SIZE, shuffle=False)

model = build_model()
optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=0.05)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=Config.LEARNING_RATE, steps_per_epoch=len(train_loader), epochs=Config.EPOCHS)
scaler = torch.amp.GradScaler('cuda') # Fixed for PyTorch 2.x

best_acc = 0.0
for epoch in range(Config.EPOCHS):
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for images, labels in pbar:
        images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
        images, labels = mix_op(images, labels)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'): # Fixed for PyTorch 2.x
            outputs = model(images); loss = criterion(outputs, labels)
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); scheduler.step()
        pbar.set_postfix({'loss': loss.item()})
    
    # Validation
    model.eval(); correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images.to(Config.DEVICE)); _, pred = outputs.max(1)
            total += labels.size(0); correct += pred.eq(labels.to(Config.DEVICE)).sum().item()
    acc = 100. * correct / total
    print(f"Valid Acc: {acc:.2f}%")
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best_model.pth")
```

---

### Optional Cell: Ultra-Fine-Tuning (Squeeze +2-3% Accuracy)
*Run this ONLY after the main training (Cell 4) finishes. It uses a tiny learning rate to perfect the model.*

```python
# 1. Load the best weights from previous training
model.load_state_dict(torch.load("best_model.pth"))

# 2. Setup very low learning rate
REFINEMENT_LR = 1e-5 
REFINEMENT_EPOCHS = 5

optimizer = optim.AdamW(model.parameters(), lr=REFINEMENT_LR, weight_decay=0.01)
criterion = nn.CrossEntropyLoss(label_smoothing=0.05) # Less smoothing for final polish
scaler = torch.amp.GradScaler('cuda')

print(f"[*] Starting Refinement for {REFINEMENT_EPOCHS} epochs...")

for epoch in range(REFINEMENT_EPOCHS):
    model.train()
    pbar = tqdm(train_loader, desc=f"Refine {epoch+1}")
    for images, labels in pbar:
        images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            outputs = model(images); loss = criterion(outputs, labels)
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        pbar.set_postfix({'loss': loss.item()})
    
    # Validation check
    acc = validate(model, val_loader)
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best_model.pth")
        print(f"--> [REFINED] New Best Accuracy: {acc:.2f}%")

print("[*] Refinement complete. Proceed to Prediction Cell.")
```

---

### Replace Section: "Generating Predections & FINAL.csv"

```python
# --- Advanced Prediction with TTA ---
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

test_ds = MLDataset(test_df, Config.IMAGE_PATH, transform=get_transforms('val'), is_test=True)
test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

all_preds = []
img_names = []

with torch.no_grad():
    for images, names in tqdm(test_loader, desc="PREDICTING"):
        outputs = model(images.to(Config.DEVICE))
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        img_names.extend(names)

# Save Final Submission
submission = pd.DataFrame({"IMAGE": img_names, "LABEL": all_preds})
submission.to_csv("FINAL.csv", index=False)
print("✅ FINAL.csv generated successfully!")
```
