import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

# --- 1. SATELLITE CONFIGURATION ---
IMG_SIZE = 224 # Upscale for EfficientNet
BATCH_SIZE = 32
EPOCHS = 20 
LEARNING_RATE = 1e-3
NUM_CLASSES = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Path handling for main_dataset
BASE_PATH = \"/kaggle/input/neural-nexus-internal-round/DATASET\" 
if not os.path.exists(BASE_PATH):
    BASE_PATH = \"./main_dataset/DATASET\"

# --- 2. DATASET & AUGMENTATION ---
class ImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, is_test=False, label_map=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        self.label_map = label_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # main_dataset uses IMAGE column
        img_name = self.df.iloc[idx, 0] 
        img_path = os.path.join(self.root_dir, img_name)
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE))
        if self.transform: image = self.transform(image)
        if self.is_test: return image
        
        # main_dataset uses LABEL column
        label_name = self.df.iloc[idx, 1]
        label = self.label_map[label_name] if self.label_map else int(label_name)
        return image, label

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(180),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def train():
    print(f"Professional PyTorch Pipeline | Dataset: main_dataset | Device: {DEVICE}")
    
    # Load Training Data
    train_csv = os.path.join(BASE_PATH, 'Train.csv')
    train_df = pd.read_csv(train_csv)
    unique_labels = sorted(train_df['LABEL'].unique())
    label_map = {name: i for i, name in enumerate(unique_labels)}
    inv_label_map = {i: name for name, i in label_map.items()}
    print(f"Mapping for {NUM_CLASSES} classes: {label_map}")

    # Dataset locations
    train_img_dir = os.path.join(BASE_PATH, 'train_images')
    test_img_dir = os.path.join(BASE_PATH, 'test_images')

    # DataLoaders
    # Using 80/20 split for validation if no separate file exists, 
    # but the user has main_dataset/DATASET/Test.csv... wait, Test.csv doesn't have labels.
    # We should split Train.csv for validation.
    from sklearn.model_selection import train_test_split
    train_data, val_data = train_test_split(train_df, test_size=0.15, stratify=train_df['LABEL'], random_state=42)
    
    # Save temp split files
    train_data.to_csv('temp_train.csv', index=False)
    val_data.to_csv('temp_val.csv', index=False)

    train_ds = ImageDataset('temp_train.csv', train_img_dir, transform=train_transform, label_map=label_map)
    val_ds = ImageDataset('temp_val.csv', train_img_dir, transform=val_transform, label_map=label_map)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Build Model
    model = models.efficientnet_b0(weights='DEFAULT')
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    model = model.to(DEVICE)

    # Optimization
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, steps_per_epoch=len(train_loader), epochs=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_f1 = 0.0
    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            pbar.set_postfix({'loss': f"{train_loss/(pbar.n+1):.4f}", 'acc': f"{100*train_correct/train_total:.2f}%"})

        # Validation with F1-Score
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        acc = accuracy_score(all_labels, all_preds) * 100
        f1 = f1_score(all_labels, all_preds, average='macro') * 100
        print(f"⭐ Validation Acc: {acc:.2f}% | F1-Score: {f1:.2f}%")
        
        if f1 > best_f1:
            best_f1 = f1
            torch.save({'model': model.state_dict(), 'inv_map': inv_label_map}, 'best_model.pth')
            print(f"🔥 New Best Model Saved (Best F1: {f1:.2f}%)")

    # Final Inference
    print("\n🏁 Generating FINAL.csv...")
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model'])
    inv_map = checkpoint['inv_map']
    model.eval()
    
    test_csv_path = os.path.join(BASE_PATH, 'Test.csv')
    test_ds = ImageDataset(test_csv_path, test_img_dir, val_transform, is_test=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    predictions = []
    with torch.no_grad():
        for images in tqdm(test_loader, desc="Inference"):
            outputs = model(images.to(DEVICE))
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())

    # FINAL.csv with Integer Labels
    pd.DataFrame({
        'IMAGE': pd.read_csv(test_csv_path)['IMAGE'], 
        'LABEL': predictions
    }).to_csv('FINAL.csv', index=False)
    print(f"✅ DONE! Created FINAL.csv with {best_f1:.2f}% F1 target achieved.")
    
    # Cleanup
    if os.path.exists('temp_train.csv'): os.remove('temp_train.csv')
    if os.path.exists('temp_val.csv'): os.remove('temp_val.csv')

if __name__ == "__main__":
    train()
