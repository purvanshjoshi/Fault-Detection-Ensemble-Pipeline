import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.transforms import v2
from PIL import Image
from tqdm.auto import tqdm
import numpy as np
import glob

# ==========================================
# MASTER CONFIGURATION (Dynamic Detection)
# ==========================================
class Config:
    # We dynamically search for 'ML FINAL DATASET' in Kaggle input
    try:
        # SEARCH: Matches /kaggle/input/*/ML FINAL DATASET
        search_path = glob.glob("/kaggle/input/*/ML FINAL DATASET")
        if search_path:
            DATA_DIR = search_path[0]
        else:
            # Fallback for manual uploads
            DATA_DIR = "/kaggle/input/alrieeena26-ml-challenge-by-ieee-sb-gehu/ML FINAL DATASET"
    except:
        DATA_DIR = "./ML FINAL DATASET" # Local fallback

    IMAGE_PATH = os.path.join(DATA_DIR, "images")
    TRAIN_CSV = os.path.join(DATA_DIR, "TRAIN.csv")
    TEST_CSV = os.path.join(DATA_DIR, "TEST.csv")
    
    # Noble Model: ConvNeXt-Tiny (Modern CNN)
    MODEL_NAME = "convnext_tiny" 
    BATCH_SIZE = 32
    NUM_CLASSES = 397
    LEARNING_RATE = 2e-4
    EPOCHS = 15
    VAL_SPLIT = 0.2
    IMAGE_SIZE = 224
    
    # Advanced Features
    MIXED_PRECISION = True
    USE_MIXUP_CUTMIX = True
    TTA_STEPS = 5 
    LABEL_SMOOTHING = 0.1
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = os.cpu_count()

print(f"[*] Data Directory Detected: {Config.DATA_DIR}")

# ==========================================
# ADVANCED DATA TRANSFORMS
# ==========================================
def get_transforms(phase='train'):
    if phase == 'train':
        return v2.Compose([
            v2.RandomResizedCrop(size=(Config.IMAGE_SIZE, Config.IMAGE_SIZE), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandAugment(num_ops=2, magnitude=12),
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return v2.Compose([
        v2.Resize(size=(Config.IMAGE_SIZE, Config.IMAGE_SIZE), antialias=True),
        v2.CenterCrop(size=(Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_tta_transforms():
    return v2.Compose([
        v2.RandomResizedCrop(size=(Config.IMAGE_SIZE, Config.IMAGE_SIZE), antialias=True, scale=(0.9, 1.0)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Mixup/Cutmix
cutmix = v2.CutMix(num_classes=Config.NUM_CLASSES)
mixup = v2.MixUp(num_classes=Config.NUM_CLASSES)
mix_op = v2.RandomChoice([cutmix, mixup])

# ==========================================
# DATASET CLASS
# ==========================================
class MLDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, is_test=False):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['IMAGE'])
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = Image.new("RGB", (Config.IMAGE_SIZE, Config.IMAGE_SIZE))
        if self.transform: image = self.transform(image)
        if self.is_test: return image, row['IMAGE']
        return image, int(row['LABEL'])

# ==========================================
# MODEL, TRAINER & INFERENCE
# ==========================================
def build_model():
    print(f"[MODEL] Building {Config.MODEL_NAME}...")
    model = models.convnext_tiny(weights="IMAGENET1K_V1")
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, Config.NUM_CLASSES)
    return model.to(Config.DEVICE)

def train_model(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=0.05)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=Config.LEARNING_RATE, steps_per_epoch=len(train_loader), epochs=Config.EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=Config.MIXED_PRECISION)

    best_acc = 0.0
    for epoch in range(Config.EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")
        for images, labels in pbar:
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            if Config.USE_MIXUP_CUTMIX: images, labels = mix_op(images, labels)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=Config.MIXED_PRECISION):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            pbar.set_postfix({'loss': loss.item()})
        
        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images.to(Config.DEVICE)); _, pred = outputs.max(1)
                total += labels.size(0); correct += pred.eq(labels.to(Config.DEVICE)).sum().item()
        acc = 100. * correct / total
        print(f"Valid Acc: {acc:.2f}%")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_model.pth")

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    train_full = pd.read_csv(Config.TRAIN_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)
    
    val_size = int(len(train_full) * Config.VAL_SPLIT)
    val_df = train_full.sample(n=val_size, random_state=42)
    train_df = train_full.drop(val_df.index)
    
    train_loader = DataLoader(MLDataset(train_df, Config.IMAGE_PATH, get_transforms('train')), batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)
    val_loader = DataLoader(MLDataset(val_df, Config.IMAGE_PATH, get_transforms('val')), batch_size=Config.BATCH_SIZE, shuffle=False)
    
    model = build_model()
    train_model(model, train_loader, val_loader)
    
    print("[INFER] Generating predictions with TTA...")
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    test_loader = DataLoader(MLDataset(test_df, Config.IMAGE_PATH, get_transforms('test'), is_test=True), batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # Simplified TTA (Horizontal Flip)
    all_probs = []
    names = []
    with torch.no_grad():
        for images, img_names in tqdm(test_loader):
            outputs = torch.softmax(model(images.to(Config.DEVICE)), dim=1)
            all_probs.append(outputs.cpu().numpy())
            names.extend(img_names)
            
    final_labels = np.argmax(np.concatenate(all_probs), axis=1)
    pd.DataFrame({"IMAGE": names, "LABEL": final_labels}).to_csv("FINAL.csv", index=False)
    print("DONE! FINAL.csv saved.")
