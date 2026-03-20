"""
Local Fast-Track Training Script for alrIEEEna26 ML Challenge.
Uses MobileNetV3-Small for rapid training on local hardware.
"""
# pylint: disable=import-error
import os
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm.auto import tqdm

# pylint: disable=too-few-public-methods
class Config:
    """
    Configuration parameters for local training.
    """
    DATA_DIR = "d:/ML_Arena/ML FINAL DATASET"
    IMAGE_PATH = os.path.join(DATA_DIR, "images")
    TRAIN_CSV = os.path.join(DATA_DIR, "TRAIN.csv")
    TEST_CSV = os.path.join(DATA_DIR, "TEST.csv")

    # Modern Fast Settings
    MODEL_NAME = "mobilenet_v3_small"
    NUM_CLASSES = 397
    BATCH_SIZE = 64
    EPOCHS = 5
    IMAGE_SIZE = 128
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Basic Fast Transform
fast_transform = transforms.Compose([
    transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class FastDataset(Dataset):
    """
    Dataset class for loading images from the local file system.
    """
    def __init__(self, dataset_df, is_test=False):
        self.df = dataset_df
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            img_name = row['IMAGE']
            full_path = os.path.join(Config.IMAGE_PATH, img_name)
            image = Image.open(full_path).convert("RGB")
            image = fast_transform(image)
        except (IOError, OSError):
            image = torch.zeros(3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)

        if self.is_test:
            return image, row['IMAGE']
        return image, int(row['LABEL'])

# pylint: disable=too-many-locals
def run_local_training():
    """
    Main training and inference execution loop.
    """
    print(f"[*] Starting Modern Local Training on {Config.DEVICE}")
    print(f"[*] Data: {Config.DATA_DIR}")

    train_full = pd.read_csv(Config.TRAIN_CSV)
    train_ds = FastDataset(train_full)
    loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0)

    print(f"[*] Building {Config.MODEL_NAME}...")
    model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, Config.NUM_CLASSES)
    model = model.to(Config.DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Training Loop
    for epoch in range(Config.EPOCHS):
        model.train()
        pbar = tqdm(loader, desc=f"Local Epoch {epoch+1}")
        for images, labels in pbar:
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Generate Local FINAL.csv
    print("[*] Generating local FINAL_LOCAL.csv...")
    model.eval()
    test_df = pd.read_csv(Config.TEST_CSV)
    test_ds = FastDataset(test_df, is_test=True)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE)
    results = []
    with torch.no_grad():
        for images, names in tqdm(test_loader):
            outputs = model(images.to(Config.DEVICE))
            _, pred = outputs.max(1)
            for n, l in zip(names, pred.cpu().numpy()):
                results.append({"IMAGE": n, "LABEL": int(l)})

    output_path = "d:/ML_Arena/ML FINAL DATASET/FINAL_LOCAL.csv"
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"✅ Local Fast-Track Submission Ready: {output_path}")

if __name__ == "__main__":
    run_local_training()
