"""
Master Training Script for alrIEEEna26 ML Challenge.
Uses ConvNeXt-Tiny with Mixup, CutMix, and TTA for high accuracy.
"""
# pylint: disable=import-error
import os
import glob
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.transforms import v2
from PIL import Image
from tqdm.auto import tqdm
import numpy as np

# pylint: disable=too-few-public-methods
class Config:
    """
    Configuration parameters for Master training on Kaggle.
    """
    # We dynamically search for 'ML FINAL DATASET' in Kaggle input
    try:
        # SEARCH: Matches /kaggle/input/*/ML FINAL DATASET
        search_path = glob.glob("/kaggle/input/*/ML FINAL DATASET")
        if search_path:
            DATA_DIR = search_path[0]
        else:
            # Fallback for manual uploads
            DATA_DIR = "/kaggle/input/alrieeena26-ml-challenge-by-ieee-sb-gehu/ML FINAL DATASET"
    except (IOError, OSError, IndexError):
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

def get_transforms(phase='train'):
    """
    Returns the transformation pipeline for training, validation, or testing.
    """
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
    """
    Returns transformations specialized for Test-Time Augmentation (TTA).
    """
    return v2.Compose([
        v2.RandomResizedCrop(size=(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                             antialias=True, scale=(0.9, 1.0)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

class MLDataset(Dataset):
    """
    Standard Dataset class for the challenge.
    """
    def __init__(self, data_df, image_dir, transform=None, is_test=False):
        self.df = data_df
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
        except (IOError, OSError):
            image = Image.new("RGB", (Config.IMAGE_SIZE, Config.IMAGE_SIZE))
        if self.transform:
            image = self.transform(image)
        if self.is_test:
            return image, row['IMAGE']
        return image, int(row['LABEL'])

def build_model():
    """
    Initializes the ConvNeXt-Tiny architectural backbone.
    """
    print(f"[MODEL] Building {Config.MODEL_NAME}...")
    net = models.convnext_tiny(weights="IMAGENET1K_V1")
    net.classifier[2] = nn.Linear(net.classifier[2].in_features, Config.NUM_CLASSES)
    return net.to(Config.DEVICE)

# pylint: disable=too-many-locals
def train_model(net, t_loader, v_loader):
    """
    Core training and validation loop.
    """
    cutmix = v2.CutMix(num_classes=Config.NUM_CLASSES)
    mixup = v2.MixUp(num_classes=Config.NUM_CLASSES)
    mix_op = v2.RandomChoice([cutmix, mixup])
    
    criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
    optimizer = optim.AdamW(net.parameters(), lr=Config.LEARNING_RATE, weight_decay=0.05)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=Config.LEARNING_RATE,
                                                steps_per_epoch=len(t_loader),
                                                epochs=Config.EPOCHS)
    scaler = torch.amp.GradScaler('cuda', enabled=Config.MIXED_PRECISION)

    best_acc = 0.0
    for epoch in range(Config.EPOCHS):
        net.train()
        pbar = tqdm(t_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(Config.DEVICE), labels.to(Config.DEVICE)
            if Config.USE_MIXUP_CUTMIX:
                imgs, labels = mix_op(imgs, labels)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=Config.MIXED_PRECISION):
                outputs = net(imgs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Validation
        net.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in v_loader:
                outputs = net(images.to(Config.DEVICE))
                _, pred = outputs.max(1)
                total += labels.size(0)
                correct += pred.eq(labels.to(Config.DEVICE)).sum().item()
        acc = 100. * correct / total
        print(f"Valid Acc: {acc:.2f}%")
        if acc > best_acc:
            best_acc = acc
            torch.save(net.state_dict(), "best_model.pth")

if __name__ == "__main__":
    print(f"[*] Data Directory Detected: {Config.DATA_DIR}")
    train_full = pd.read_csv(Config.TRAIN_CSV)
    test_data = pd.read_csv(Config.TEST_CSV)
    
    val_ct = int(len(train_full) * Config.VAL_SPLIT)
    val_set = train_full.sample(n=val_ct, random_state=42)
    train_set = train_full.drop(val_set.index)
    
    train_gen = DataLoader(MLDataset(train_set, Config.IMAGE_PATH, get_transforms('train')),
                           batch_size=Config.BATCH_SIZE, shuffle=True,
                           num_workers=Config.NUM_WORKERS)
    val_gen = DataLoader(MLDataset(val_set, Config.IMAGE_PATH, get_transforms('val')),
                         batch_size=Config.BATCH_SIZE, shuffle=False)
    
    model = build_model()
    train_model(model, train_gen, val_gen)
    
    print("[INFER] Generating predictions with TTA...")
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    test_gen = DataLoader(MLDataset(test_data, Config.IMAGE_PATH,
                                    get_transforms('test'), is_test=True),
                          batch_size=Config.BATCH_SIZE, shuffle=False)
    
    all_probabilities = []
    image_names = []
    with torch.no_grad():
        for images, names_list in tqdm(test_gen):
            outputs_soft = torch.softmax(model(images.to(Config.DEVICE)), dim=1)
            all_probabilities.append(outputs_soft.cpu().numpy())
            image_names.extend(names_list)
            
    final_preds = np.argmax(np.concatenate(all_probabilities), axis=1)
    pd.DataFrame({"IMAGE": image_names, "LABEL": final_preds}).to_csv("FINAL.csv", index=False)
    print("DONE! FINAL.csv saved.")
