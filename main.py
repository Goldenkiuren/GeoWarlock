import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image, ImageFile
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

# --- HARDWARE OPTIMIZATIONS (RTX 4080 SPECIFIC) ---
# 1. Enable TF32 for faster math on Ampere/Ada GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# 2. Enable cuDNN benchmarking to pick the fastest convolution algos
torch.backends.cudnn.benchmark = True

# --- CONFIGURATION ---
BATCH_SIZE = 128
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
USE_SPATIAL_SPLIT = True 

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Paths
DB_SUBPATH = os.path.join('database', 'images') 
SEQ_INFO_FILE = os.path.join('database', 'seq_info.csv')
RAW_DATA_FILE = os.path.join('database', 'raw.csv')

class CityClassificationDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception:
            return torch.zeros((3, IMG_SIZE, IMG_SIZE)), label

def create_splits(root_dir, val_size=0.2):
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    
    classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    print(f"Found {len(classes)} cities. Using Spatial Split: {USE_SPATIAL_SPLIT}")

    for city in classes:
        city_path = os.path.join(root_dir, city)
        images_dir = os.path.join(city_path, DB_SUBPATH)
        seq_path = os.path.join(city_path, SEQ_INFO_FILE)
        raw_path = os.path.join(city_path, RAW_DATA_FILE)
        
        if not os.path.exists(images_dir):
            continue

        valid_exts = ('.jpg', '.jpeg', '.png')
        key_to_path = {os.path.splitext(f)[0]: os.path.join(images_dir, f) 
                       for f in os.listdir(images_dir) if f.lower().endswith(valid_exts)}
        
        # Strategy 1: Spatial Split
        if USE_SPATIAL_SPLIT and os.path.exists(raw_path):
            try:
                df_raw = pd.read_csv(raw_path)
                df_raw = df_raw[df_raw['key'].isin(key_to_path.keys())]
                coords = df_raw[['lat', 'lon']].values
                kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)
                df_raw['cluster'] = kmeans.fit_predict(coords)
                
                val_cluster_id = 0 
                train_keys = df_raw[df_raw['cluster'] != val_cluster_id]['key'].tolist()
                val_keys = df_raw[df_raw['cluster'] == val_cluster_id]['key'].tolist()
                
                print(f"  {city} (Spatial): Train {len(train_keys)} | Val {len(val_keys)}")
                for k in train_keys: train_paths.append(key_to_path[k]); train_labels.append(class_to_idx[city])
                for k in val_keys: val_paths.append(key_to_path[k]); val_labels.append(class_to_idx[city])
                continue 
            except Exception:
                pass 

        # Strategy 2: Sequence Split (Fallback)
        if os.path.exists(seq_path):
            df = pd.read_csv(seq_path)
            df = df[df['key'].isin(key_to_path.keys())]
            if 'sequence_key' in df.columns:
                seq_groups = df.groupby('sequence_key')['key'].apply(list).tolist()
                train_seqs, val_seqs = train_test_split(seq_groups, test_size=val_size, random_state=42)
                for seq in train_seqs:
                    for k in seq: train_paths.append(key_to_path[k]); train_labels.append(class_to_idx[city])
                for seq in val_seqs:
                    for k in seq: val_paths.append(key_to_path[k]); val_labels.append(class_to_idx[city])
                print(f"  {city} (Sequence): Train {len(train_seqs)} seqs | Val {len(val_seqs)} seqs")

    return train_paths, train_labels, val_paths, val_labels, len(classes)

class ViTForCityClassification(nn.Module):
    def __init__(self, num_classes):
        super(ViTForCityClassification, self).__init__()
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        input_dim = self.vit.heads.head.in_features
        self.vit.heads = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.vit(x)

def main():
    dataset_root = "data" 
    
    print("Preparing Robust Splits...")
    train_paths, train_y, val_paths, val_y, num_classes = create_splits(dataset_root)
    print(f"Total: {len(train_paths)} Train, {len(val_paths)} Val")

    # Transforms (kept same)
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = CityClassificationDataset(train_paths, train_y, transform=train_transform)
    val_dataset = CityClassificationDataset(val_paths, val_y, transform=val_transform)

    # 3. DATALOADER OPTIMIZATION
    # persistent_workers=True keeps the RAM loaded between epochs
    # prefetch_factor buffers batches so GPU never waits for CPU
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=8, 
        pin_memory=True,
        persistent_workers=True, 
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=8, 
        pin_memory=True,
        persistent_workers=True, 
        prefetch_factor=2
    )

    model = ViTForCityClassification(num_classes=num_classes).to(DEVICE)
    
    # 4. MODEL COMPILATION (The Big Speedup)
    # This fuses layers and optimizes for the 4080 specifically
    print("Compiling model (this takes a minute at start)...")
    model = torch.compile(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scaler = torch.amp.GradScaler('cuda')
    
    print(f"Starting Training on {DEVICE}...")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss, correct, total = 0, 0, 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for images, labels in loop:
            # non_blocking allows async data transfer
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loop.set_postfix(acc=100 * correct / total)
        
        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        print(f"Epoch {epoch+1}: Train Acc: {100*correct/total:.1f}% | Val Acc: {100*val_correct/val_total:.1f}%")
        torch.save(model.state_dict(), "best_city_model.pth")

if __name__ == "__main__":
    main()