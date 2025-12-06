import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image, ImageFile
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from collections import Counter

# --- HARDWARE OPTIMIZATIONS ---
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# --- CONFIGURATION ---
BATCH_SIZE = 128
NUM_EPOCHS = 1 
LEARNING_RATE = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
USE_SPATIAL_SPLIT = True 

ImageFile.LOAD_TRUNCATED_IMAGES = True

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
    
    print(f"Found {len(classes)} cities.")
    print("Scanning both 'database' and 'query' folders with Quality Filters...")

    SUB_FOLDERS = ['database', 'query']

    for city in classes:
        city_path = os.path.join(root_dir, city)
        
        city_image_paths = []
        city_gps_coords = [] 
        
        # Load GPS Data Lookup
        key_to_gps = {}
        # Check both subfolders for raw.csv to build a master GPS map for the city
        for sub in SUB_FOLDERS:
            raw_path = os.path.join(city_path, sub, 'raw.csv')
            if os.path.exists(raw_path):
                try:
                    df_raw = pd.read_csv(raw_path)
                    for _, row in df_raw.iterrows():
                        key_to_gps[str(row['key'])] = (row['lat'], row['lon'])
                except:
                    pass

        # Collect Images
        for sub in SUB_FOLDERS:
            base_path = os.path.join(city_path, sub)
            images_dir = os.path.join(base_path, 'images')
            meta_path = os.path.join(base_path, 'postprocessed.csv')
            
            if not os.path.exists(images_dir):
                continue
            
            # Load Filter
            allowed_keys = None
            if os.path.exists(meta_path):
                try:
                    df_meta = pd.read_csv(meta_path)
                    clean_df = df_meta[(df_meta['night'] == False) & (df_meta['control_panel'] == False)]
                    allowed_keys = set(clean_df['key'].astype(str))
                except:
                    pass 

            valid_exts = ('.jpg', '.jpeg', '.png')
            for f in os.listdir(images_dir):
                if f.lower().endswith(valid_exts):
                    key = os.path.splitext(f)[0]
                    if allowed_keys is not None and key not in allowed_keys:
                        continue 
                    
                    full_path = os.path.join(images_dir, f)
                    city_image_paths.append(full_path)
                    
                    # Get GPS
                    if key in key_to_gps:
                        city_gps_coords.append(key_to_gps[key])
                    else:
                        city_gps_coords.append((0.0, 0.0))

        if not city_image_paths:
            continue

        print(f"  {city}: {len(city_image_paths)} images (Unique GPS: {len(np.unique(city_gps_coords, axis=0))})")
        
        # Check if we have enough UNIQUE coordinates to actually cluster
        coords = np.array(city_gps_coords)
        unique_coords = np.unique(coords, axis=0)
        
        # We need at least 5 unique locations to do a 5-cluster split
        can_use_spatial = USE_SPATIAL_SPLIT and len(unique_coords) >= 5

        if can_use_spatial:
            try:
                kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)
                clusters = kmeans.fit_predict(coords)
                
                # Use cluster 0 for validation
                val_cluster_id = 0
                
                for i, path in enumerate(city_image_paths):
                    if clusters[i] == val_cluster_id:
                        val_paths.append(path)
                        val_labels.append(class_to_idx[city])
                    else:
                        train_paths.append(path)
                        train_labels.append(class_to_idx[city])
            except Exception as e:
                print(f"  [Warning] {city}: Spatial split error ({e}). Falling back to random.")
                can_use_spatial = False # Trigger fallback below

        # Strategy 2: Random Split (Fallback)
        if not can_use_spatial:
            # print(f"  Note: Using Random Split for {city} (insufficient GPS data)")
            tr_paths, va_paths = train_test_split(city_image_paths, test_size=val_size, random_state=42)
            for p in tr_paths:
                train_paths.append(p)
                train_labels.append(class_to_idx[city])
            for p in va_paths:
                val_paths.append(p)
                val_labels.append(class_to_idx[city])

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
    
    print("Preparing Clean Splits (Database + Query)...")
    train_paths, train_y, val_paths, val_y, num_classes = create_splits(dataset_root)
    print(f"Total Valid Images: {len(train_paths)} Train, {len(val_paths)} Val")

    # Balancing Weights
    print("Computing Class Weights...")
    class_counts = Counter(train_y)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in train_y]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    # Aggressive Augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.4, 1.0)), 
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
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

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler, 
        num_workers=12, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )

    model = ViTForCityClassification(num_classes=num_classes).to(DEVICE)
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