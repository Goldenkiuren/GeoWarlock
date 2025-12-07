import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
import torch.nn.functional as F

# --- CONFIGURAÇÃO ---
TEST_FOLDER = "test"        
DATASET_ROOT = "data"       
MODEL_PATH = "best_city_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Keep your mappings exactly the same
PREFIX_MAP = {
    'sp': 'saopaulo', 'ba': 'buenosaires', 'am': 'amsterdam', 'bk': 'bangkok',
    'he': 'helsinki', 'kp': 'kampala', 'me': 'melbourne', 'mi': 'miami',
    'sf': 'sf', 'tk': 'tokyo', 'go': 'goa', 'ma': 'manila', 'at': 'athens',
    'pa': 'paris', 'bu': 'budapest', 'ot': 'ottawa', 'au': 'austin',
    'be': 'bengaluru', 'br': 'berlin', 'bo': 'boston', 'co': 'copenhagen',
    'lo': 'london', 'mo': 'moscow', 'ph': 'phoenix', 'st': 'stockholm',
    'tr': 'trondheim', 'to': 'toronto', 'zu': 'zurich'
}

CITY_TO_CONTINENT = {
    'saopaulo': 'South America', 'buenosaires': 'South America',
    'miami': 'North America', 'sf': 'North America', 'ottawa': 'North America',
    'austin': 'North America', 'boston': 'North America', 'phoenix': 'North America',
    'toronto': 'North America', 'amsterdam': 'Europe', 'helsinki': 'Europe',
    'athens': 'Europe', 'paris': 'Europe', 'budapest': 'Europe',
    'berlin': 'Europe', 'copenhagen': 'Europe', 'london': 'Europe',
    'moscow': 'Europe', 'stockholm': 'Europe', 'trondheim': 'Europe',
    'zurich': 'Europe', 'bangkok': 'Asia', 'tokyo': 'Asia', 'goa': 'Asia',
    'manila': 'Asia', 'bengaluru': 'Asia', 'kampala': 'Africa',
    'melbourne': 'Oceania'
}
IGNORED_CITIES = ['amman', 'nairobi']

def get_classes(dataset_root):
    classes = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
    classes.sort()
    classes = [c for c in classes if c not in IGNORED_CITIES]
    return classes

class ViTForCityClassification(nn.Module):
    def __init__(self, num_classes):
        super(ViTForCityClassification, self).__init__()
        self.vit = models.vit_b_16(weights=None) 
        input_dim = self.vit.heads.head.in_features
        self.vit.heads = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.vit(x)

def predict_tta(model, image, transform_base):
    """
    Test Time Augmentation: Run prediction on multiple variations of the image
    and average the probabilities.
    """
    # Create 5 variations
    transforms_tta = [
        # 1. Original
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        # 2. Flip
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        # 3. Center Crop (Zoom)
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    ]

    preds = []
    with torch.no_grad():
        for t in transforms_tta:
            img_tensor = t(image).unsqueeze(0).to(DEVICE)
            output = model(img_tensor)
            probs = F.softmax(output, dim=1)
            preds.append(probs)
    
    # Average predictions
    avg_probs = torch.stack(preds).mean(dim=0)
    return avg_probs

def main():
    classes = get_classes(DATASET_ROOT)
    num_classes = len(classes)
    print(f"Loaded {num_classes} classes.")

    model = ViTForCityClassification(num_classes=num_classes).to(DEVICE)
    print(f"Loading model from {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    
    new_state_dict = {}
    for key, value in checkpoint.items():
        if key.startswith("_orig_mod."):
            new_key = key.replace("_orig_mod.", "")
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
            
    model.load_state_dict(new_state_dict)
    model.eval()
    
    # Base transform is handled inside the TTA function now
    
    stats = {code: {'1st': 0, '2nd': 0, '3rd': 0, 'miss': 0, 'total': 0} for code in PREFIX_MAP.keys()}
    continent_stats = {}
    
    print(f"\nScanning folder '{TEST_FOLDER}'...")
    valid_files = [f for f in os.listdir(TEST_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not valid_files:
        print("No images found!")
        return

    print(f"Found {len(valid_files)} images. Running TTA inference...\n")

    for fname in valid_files:
        prefix = fname[:2].lower()
        if prefix not in PREFIX_MAP: continue
            
        true_class_name = PREFIX_MAP[prefix]
        true_continent = CITY_TO_CONTINENT.get(true_class_name, "Unknown")
        
        if true_continent not in continent_stats:
            continent_stats[true_continent] = {'1st': 0, '2nd': 0, '3rd': 0, 'miss': 0, 'total': 0}

        img_path = os.path.join(TEST_FOLDER, fname)
        try:
            image = Image.open(img_path).convert("RGB")
            
            # --- TTA PREDICTION ---
            probs = predict_tta(model, image, None)
            top3_prob, top3_idx = torch.topk(probs, 3)
            
        except Exception as e:
            print(f"Error {fname}: {e}")
            continue

        top3_names = [classes[idx] for idx in top3_idx[0]]
        
        stats[prefix]['total'] += 1
        
        # Reduced logging clutter
        if true_class_name == top3_names[0]:
            stats[prefix]['1st'] += 1
        elif true_class_name == top3_names[1]:
            stats[prefix]['2nd'] += 1
        elif true_class_name == top3_names[2]:
            stats[prefix]['3rd'] += 1
        else:
            stats[prefix]['miss'] += 1
            # print(f"❌ {fname} -> {top3_names[0]} (True: {true_class_name})")

        continent_stats[true_continent]['total'] += 1
        pred_cont_1 = CITY_TO_CONTINENT.get(top3_names[0], "Unknown")
        
        if pred_cont_1 == true_continent:
            continent_stats[true_continent]['1st'] += 1
        # Simplified continent logic for brevity in this snippet
        
    # ------------------ PRINT CITY STATS ------------------
    print("\n" + "="*60)
    print(f"{'CITY':<15} | {'TOTAL':<5} | {'1st (Correct)':<13} | {'ACC':<5}")
    print("-" * 60)
    
    total_correct = 0
    total_imgs = 0
    
    for prefix, data in stats.items():
        if data['total'] == 0: continue
        city_name = PREFIX_MAP[prefix].upper()
        t = data['total']
        p1 = f"{data['1st']}"
        acc = f"{int(data['1st']/t*100)}%"
        print(f"{city_name:<15} | {t:<5} | {p1:<13} | {acc:<5}")
        total_correct += data['1st']
        total_imgs += t

    print("-" * 60)
    if total_imgs > 0:
        print(f"OVERALL ACCURACY: {total_correct}/{total_imgs} = {total_correct/total_imgs*100:.1f}%")