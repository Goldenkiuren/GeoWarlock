import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image, ImageFile
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from collections import Counter
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
BATCH_SIZE = 32      
NUM_EPOCHS = 5
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
    raw_classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    raw_classes.sort()
    final_classes = [] 
    print(f"Scanning {len(raw_classes)} potential cities...")
    SUB_FOLDERS = ['database', 'query']
    city_data_buffer = {}
    for city in raw_classes:
        city_path = os.path.join(root_dir, city)
        city_image_paths = []
        city_gps_coords = [] 
        city_sequences = [] 
        key_to_gps = {}
        key_to_seq = {} 
        for sub in SUB_FOLDERS:
            base_sub_path = os.path.join(city_path, sub)
            raw_path = os.path.join(base_sub_path, 'raw.csv')
            if os.path.exists(raw_path):
                try:
                    df_raw = pd.read_csv(raw_path)
                    for _, row in df_raw.iterrows():
                        key_to_gps[str(row['key'])] = (row['lat'], row['lon'])
                except: pass
            seq_path = os.path.join(base_sub_path, 'seq_info.csv')
            if os.path.exists(seq_path):
                try:
                    df_seq = pd.read_csv(seq_path)
                    if 'key' in df_seq.columns and 'sequence_key' in df_seq.columns:
                        for _, row in df_seq.iterrows():
                            key_to_seq[str(row['key'])] = str(row['sequence_key'])
                except: pass
        for sub in SUB_FOLDERS:
            base_path = os.path.join(city_path, sub)
            images_dir = os.path.join(base_path, 'images')
            meta_path = os.path.join(base_path, 'postprocessed.csv')
            if not os.path.exists(images_dir):
                continue
            allowed_keys = None
            if os.path.exists(meta_path):
                try:
                    df_meta = pd.read_csv(meta_path)
                    clean_df = df_meta[(df_meta['night'] == False) & (df_meta['control_panel'] == False)]
                    allowed_keys = set(clean_df['key'].astype(str))
                except: pass 
            valid_exts = ('.jpg', '.jpeg', '.png')
            for f in os.listdir(images_dir):
                if f.lower().endswith(valid_exts):
                    key = os.path.splitext(f)[0]
                    if allowed_keys is not None and key not in allowed_keys:
                        continue 
                    full_path = os.path.join(images_dir, f)
                    city_image_paths.append(full_path)
                    if key in key_to_gps:
                        city_gps_coords.append(key_to_gps[key])
                    else:
                        city_gps_coords.append((0.0, 0.0))
                    if key in key_to_seq:
                        city_sequences.append(key_to_seq[key])
                    else:
                        city_sequences.append(key) 
        if len(city_image_paths) < 200:
            print(f"  [Skipping] {city}: Too few images ({len(city_image_paths)})")
            continue
        final_classes.append(city)
        city_data_buffer[city] = (city_image_paths, city_gps_coords, city_sequences)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(final_classes)}
    print(f"Final Class List: {len(final_classes)} cities.")
    for city in final_classes:
        city_image_paths, city_gps_coords, city_sequences = city_data_buffer[city]
        coords = np.array(city_gps_coords)
        unique_coords = np.unique(coords, axis=0)
        n_unique_gps = len(unique_coords)
        unique_sequences = list(set(city_sequences))
        n_unique_seq = len(unique_sequences)
        print(f"  {city}: {len(city_image_paths)} imgs | {n_unique_gps} locs | {n_unique_seq} seqs", end="")
        can_use_spatial = USE_SPATIAL_SPLIT and n_unique_gps >= 5
        split_done = False
        if can_use_spatial:
            try:
                kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)
                clusters = kmeans.fit_predict(coords)
                print(" -> Spatial Split (GPS)")
                val_cluster_id = 0
                for i, path in enumerate(city_image_paths):
                    if clusters[i] == val_cluster_id:
                        val_paths.append(path)
                        val_labels.append(class_to_idx[city])
                    else:
                        train_paths.append(path)
                        train_labels.append(class_to_idx[city])
                split_done = True
            except:
                print(" -> GPS Split Failed. Trying Sequence...")
        if not split_done and n_unique_seq > 1:
            print(" -> Sequence Split (No GPS)")
            train_seqs, val_seqs = train_test_split(unique_sequences, test_size=val_size, random_state=42)
            train_seq_set = set(train_seqs) 
            for i, path in enumerate(city_image_paths):
                seq_id = city_sequences[i]
                if seq_id in train_seq_set:
                    train_paths.append(path)
                    train_labels.append(class_to_idx[city])
                else:
                    val_paths.append(path)
                    val_labels.append(class_to_idx[city])
            split_done = True
        if not split_done:
            print(" -> Random Split (Fallback)")
            tr_paths, va_paths = train_test_split(city_image_paths, test_size=val_size, random_state=42)
            for p in tr_paths:
                train_paths.append(p)
                train_labels.append(class_to_idx[city])
            for p in va_paths:
                val_paths.append(p)
                val_labels.append(class_to_idx[city])
    return train_paths, train_labels, val_paths, val_labels, len(final_classes)
class DinoV2FineTuning(nn.Module):
    def __init__(self, num_classes):
        super(DinoV2FineTuning, self).__init__()
        print("Loading DINOv2 (Fine-Tuning / Unfrozen)...")
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        input_dim = 768 
        self.head = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)
def main():
    dataset_root = "data" 
    print("Preparing Clean Splits (Database + Query)...")
    train_paths, train_y, val_paths, val_y, num_classes = create_splits(dataset_root)
    print(f"Total Valid Images: {len(train_paths)} Train, {len(val_paths)} Val")
    print("Computing Class Weights...")
    class_counts = Counter(train_y)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in train_y]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
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
        num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )
    model = DinoV2FineTuning(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    param_groups = [
        {
            'params': model.backbone.parameters(), 
            'lr': 5e-6  
        },
        {
            'params': model.head.parameters(), 
            'lr': 1e-4  
        }
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = torch.cuda.amp.GradScaler()
    print(f"Starting Fine-Tuning on {DEVICE}...")
    print(f"Backbone LR: 5e-6 | Head LR: 1e-4 | Batch Size: {BATCH_SIZE}")
    best_val_acc = 0.0
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss, correct, total = 0, 0, 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for images, labels in loop:
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
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
        scheduler.step()
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        epoch_val_acc = 100 * val_correct / val_total
        current_lrs = [group['lr'] for group in optimizer.param_groups]
        print(f"Epoch {epoch+1}: Train Acc: {100*correct/total:.1f}% | Val Acc: {epoch_val_acc:.1f}%")
        print(f"  -> LRs: Backbone={current_lrs[0]:.2e}, Head={current_lrs[1]:.2e}")
        save_name = f"finetuned_model_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), save_name)
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), "best_finetuned_model.pth")
            print(f"  -> New Best Model Saved: best_finetuned_model.pth")
if __name__ == "__main__":
    main()