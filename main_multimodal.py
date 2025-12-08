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
import pickle
import random

# --- HARDWARE OPTIMIZATIONS ---
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# --- CONFIGURATION ---
BATCH_SIZE = 128
NUM_EPOCHS = 15        # Kept 12 (Multimodal takes longer to converge)
LEARNING_RATE = 5e-5    # Kept lower LR for stability with Fusion
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224 
OCR_CACHE_FILE = "ocr_features_smart.pkl" 
USE_SPATIAL_SPLIT = True # Adopting logic from main.py

# Probabilidades de "Cegar" uma modalidade (Modality Dropout)
p_drop_visual = 0.15 
p_drop_text = 0.10   

ImageFile.LOAD_TRUNCATED_IMAGES = True

    # --- DATASET WITH OCR ---
class MultiModalDataset(Dataset):
    # O erro estava aqui: certifique-se que 'transform=None' está nos argumentos
    def __init__(self, image_paths, labels, ocr_data, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.ocr_data = ocr_data 
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 1. Load Image
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except:
            # Fallback for corrupt images
            image = torch.zeros((3, IMG_SIZE, IMG_SIZE))

        # 2. Load OCR Features
        if img_path in self.ocr_data:
            script_vec, text_emb = self.ocr_data[img_path]
        else:
            # Fallback (Zeros)
            script_vec = np.zeros(6, dtype=np.float32)
            text_emb = np.zeros(384, dtype=np.float32)

        # Concatenate text features (384 + 6 = 390 dims)
        combined_text = np.concatenate([script_vec, text_emb], axis=0)
        text_tensor = torch.tensor(combined_text, dtype=torch.float32)

        return image, text_tensor, label

# --- MULTI-MODAL MODEL ---
class MultiModalViT(nn.Module):
    def __init__(self, num_classes):
        super(MultiModalViT, self).__init__()
        
        # 1. Visual Backbone (ViT-B/16)
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        visual_dim = self.vit.heads.head.in_features
        self.vit.heads = nn.Identity() 
        
        # 2. Text Branch processing
        # Input: 390 (384 Sentence Embed + 6 Script Probabilities)
        self.text_process = nn.Sequential(
            nn.Linear(390, 512), 
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 3. Fusion Head
        fusion_dim = visual_dim + 512
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, img, text_feat):
        vis_feat = self.vit(img)
        txt_feat_proc = self.text_process(text_feat)
        combined = torch.cat((vis_feat, txt_feat_proc), dim=1)
        return self.classifier(combined)

# --- ROBUST SPLITTING LOGIC (Ported from main.py) ---
def create_splits(root_dir, val_size=0.2):
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    
    # 1. Identificar classes válidas
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
        
        # Load Metadata Lookups
        key_to_gps = {}
        key_to_seq = {} 

        for sub in SUB_FOLDERS:
            base_sub_path = os.path.join(city_path, sub)
            
            # Tenta ler raw.csv (GPS)
            raw_path = os.path.join(base_sub_path, 'raw.csv')
            if os.path.exists(raw_path):
                try:
                    df_raw = pd.read_csv(raw_path)
                    for _, row in df_raw.iterrows():
                        key_to_gps[str(row['key'])] = (row['lat'], row['lon'])
                except: pass

            # Tenta ler seq_info.csv (Sequência)
            seq_path = os.path.join(base_sub_path, 'seq_info.csv')
            if os.path.exists(seq_path):
                try:
                    df_seq = pd.read_csv(seq_path)
                    if 'key' in df_seq.columns and 'sequence_key' in df_seq.columns:
                        for _, row in df_seq.iterrows():
                            key_to_seq[str(row['key'])] = str(row['sequence_key'])
                except: pass

        # Collect Images
        for sub in SUB_FOLDERS:
            base_path = os.path.join(city_path, sub)
            images_dir = os.path.join(base_path, 'images')
            meta_path = os.path.join(base_path, 'postprocessed.csv')
            
            if not os.path.exists(images_dir):
                continue
            
            # Load Filter (Day/Night) - CRUCIAL FOR OCR TOO
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

        # --- MINIMUM SIZE FILTER ---
        if len(city_image_paths) < 200:
            print(f"  [Skipping] {city}: Too few images ({len(city_image_paths)})")
            continue
            
        final_classes.append(city)
        city_data_buffer[city] = (city_image_paths, city_gps_coords, city_sequences)

    class_to_idx = {cls_name: i for i, cls_name in enumerate(final_classes)}
    print(f"Final Class List: {len(final_classes)} cities.")

    # Process Splits
    for city in final_classes:
        city_image_paths, city_gps_coords, city_sequences = city_data_buffer[city]
        
        coords = np.array(city_gps_coords)
        unique_coords = np.unique(coords, axis=0)
        n_unique_gps = len(unique_coords)
        
        unique_sequences = list(set(city_sequences))
        n_unique_seq = len(unique_sequences)
        
        # Logic adapted from main.py for spatial/sequence integrity
        can_use_spatial = USE_SPATIAL_SPLIT and n_unique_gps >= 5
        split_done = False

        if can_use_spatial:
            try:
                kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)
                clusters = kmeans.fit_predict(coords)
                
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
                pass
        
        if not split_done and n_unique_seq > 1:
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
            # Fallback to random
            tr_paths, va_paths = train_test_split(city_image_paths, test_size=val_size, random_state=42)
            for p in tr_paths:
                train_paths.append(p)
                train_labels.append(class_to_idx[city])
            for p in va_paths:
                val_paths.append(p)
                val_labels.append(class_to_idx[city])

    return train_paths, train_labels, val_paths, val_labels, len(final_classes)

def main():
    print("--- PHASE 4: MULTI-MODAL TRAINING (Optimized & Spatial-Aware) ---")
    
    # 1. Load OCR Data
    if not os.path.exists(OCR_CACHE_FILE):
        print(f"ERROR: {OCR_CACHE_FILE} not found. Run optimized_ocr_pipeline.py first!")
        return
    
    print(f"Loading OCR Features from {OCR_CACHE_FILE}...")
    with open(OCR_CACHE_FILE, 'rb') as f:
        ocr_data = pickle.load(f)
    print(f"OCR Data Loaded: {len(ocr_data)} entries.")

    # 2. Create Splits (Using Robust logic from main.py)
    train_paths, train_y, val_paths, val_y, num_classes = create_splits("data")
    print(f"Total Valid Images: {len(train_paths)} Train, {len(val_paths)} Val")

    # 3. Integrity Check (Crucial: Matches Paths from Split to Paths in OCR Cache)
    hits = 0
    check_limit = 1000
    for i in range(min(len(train_paths), check_limit)):
        if train_paths[i] in ocr_data: hits += 1
    if hits == 0: 
        print("CRITICAL WARNING: No OCR paths matched! Check absolute vs relative paths.")
    else:
        print(f"Integrity OK: {hits}/{min(len(train_paths), check_limit)} checked paths have OCR data.")

    # 4. Weights & Sampler (From main.py - Hard Balancing)
    print("Computing Class Weights (Hard Balancing - 1/Count)...")
    class_counts = Counter(train_y)
    weights = [1.0 / class_counts[y] for y in train_y]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    # 5. Transforms (Augmentations from main.py)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.4, 1.0)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 6. Loaders (Optimized params from main.py)
    train_ds = MultiModalDataset(train_paths, train_y, ocr_data, transform=train_tf)
    val_ds = MultiModalDataset(val_paths, val_y, ocr_data, transform=val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler, 
        num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )

    # 7. Model Setup
    print(f"Initializing MultiModal Model for {num_classes} cities...")
    model = MultiModalViT(num_classes).to(DEVICE)
    
    try:
        model = torch.compile(model)
        print("Torch Compile Enabled.")
    except: pass
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3) # Adjusted weight decay to match main.py
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = torch.amp.GradScaler('cuda')

    print("Starting Training...")
    best_acc = 0.0

    # --- WARMUP (Specific to Multimodal) ---
    print("❄️  Warmup: Freezing ViT backbone for Epoch 1...")
    try:
        if hasattr(model, '_orig_mod'): backbone = model._orig_mod.vit
        else: backbone = model.vit
        for param in backbone.parameters():
            param.requires_grad = False
    except:
        print("Could not freeze backbone, skipping warmup freeze.")

    for epoch in range(NUM_EPOCHS):
        # Unfreeze logic
        if epoch == 1:
            print("🔥 Unfreezing ViT backbone for full fine-tuning...")
            try:
                if hasattr(model, '_orig_mod'): backbone = model._orig_mod.vit
                else: backbone = model.vit
                for param in backbone.parameters():
                    param.requires_grad = True
            except: pass

        model.train()
        correct, total = 0, 0
        loop = tqdm(train_loader, desc=f"Ep {epoch+1}/{NUM_EPOCHS}")
        
        for img, txt, label in loop:
            img, txt, label = img.to(DEVICE, non_blocking=True), txt.to(DEVICE, non_blocking=True), label.to(DEVICE, non_blocking=True)
            
            # --- MODALITY DROPOUT (Keep this! It's vital for Fusion) ---
            if epoch > 0: 
                if np.random.rand() < p_drop_visual:
                    img = torch.zeros_like(img)
                elif np.random.rand() < p_drop_text:
                    txt = torch.zeros_like(txt)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                out = model(img, txt)
                loss = criterion(out, label)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            _, pred = torch.max(out, 1)
            total += label.size(0)
            correct += (pred == label).sum().item()
            loop.set_postfix(acc=100*correct/total, loss=loss.item())
            
        scheduler.step()
        
        # Validation
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for img, txt, label in val_loader:
                img, txt, label = img.to(DEVICE, non_blocking=True), txt.to(DEVICE, non_blocking=True), label.to(DEVICE, non_blocking=True)
                out = model(img, txt)
                _, pred = torch.max(out, 1)
                v_total += label.size(0)
                v_correct += (pred == label).sum().item()
        
        val_acc = 100 * v_correct / v_total
        print(f"Ep {epoch+1} Val Acc: {val_acc:.2f}%")
        
        # --- SAVE EVERY EPOCH (From main.py) ---
        save_name = f"multimodal_model_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), save_name)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_multimodal_model.pth")
            print(f"  -> Saved Best Model (Acc: {best_acc:.2f}%)")

if __name__ == "__main__":
    main()