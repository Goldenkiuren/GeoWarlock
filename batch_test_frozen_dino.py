import os
import torch
import pandas as pd
import argparse
from PIL import Image
from torchvision import transforms
import torch.nn as nn
TEST_FOLDER = "test"        
DATASET_ROOT = "data"       
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PREFIX_MAP = {
    'sp': 'saopaulo',
    'ba': 'buenosaires',
    'am': 'amsterdam',
    'bk': 'bangkok',
    'he': 'helsinki',
    'kp': 'kampala',
    'me': 'melbourne',
    'mi': 'miami',
    'sf': 'sf',
    'tk': 'tokyo',
    'go': 'goa',
    'ma': 'manila',
    'at': 'athens',
    'pa': 'paris',
    'bu': 'budapest',
    'ot': 'ottawa',
    'au': 'austin',
    'be': 'bengaluru',
    'br': 'berlin',
    'bo': 'boston',
    'co': 'cph',
    'lo': 'london',
    'mo': 'moscow',
    'ph': 'phoenix',
    'st': 'stockholm',
    'tr': 'trondheim',
    'to': 'toronto',
    'zu': 'zurich'
}
CITY_TO_CONTINENT = {
    'saopaulo': 'South America',
    'buenosaires': 'South America',
    'miami': 'North America',
    'sf': 'North America',
    'ottawa': 'North America',
    'austin': 'North America',
    'boston': 'North America',
    'phoenix': 'North America',
    'toronto': 'North America',
    'amsterdam': 'Europe',
    'helsinki': 'Europe',
    'athens': 'Europe',
    'paris': 'Europe',
    'budapest': 'Europe',
    'berlin': 'Europe',
    'cph': 'Europe',
    'london': 'Europe',
    'moscow': 'Europe',
    'stockholm': 'Europe',
    'trondheim': 'Europe',
    'zurich': 'Europe',
    'bangkok': 'Asia',
    'tokyo': 'Asia',
    'goa': 'Asia',
    'manila': 'Asia',
    'bengaluru': 'Asia',
    'kampala': 'Africa',
    'melbourne': 'Oceania'
}
IGNORED_CITIES = ['amman', 'nairobi']
def get_classes(dataset_root):
    classes = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
    classes.sort()
    classes = [c for c in classes if c not in IGNORED_CITIES]
    return classes
class DinoV2ForCityClassification(nn.Module):
    def __init__(self, num_classes):
        super(DinoV2ForCityClassification, self).__init__()
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="best_models/frozen_dinov2_best.pth")
    args = parser.parse_args()
    classes = get_classes(DATASET_ROOT)
    num_classes = len(classes)
    print(f"Loaded {num_classes} classes.")
    model = DinoV2ForCityClassification(num_classes=num_classes).to(DEVICE)
    if not os.path.exists(args.model):
        print(f"Model {args.model} not found.")
        return
    print(f"Loading weights from {args.model}...")
    checkpoint = torch.load(args.model, map_location=DEVICE)
    new_state_dict = {}
    for key, value in checkpoint.items():
        new_key = key.replace("_orig_mod.", "")
        new_state_dict[new_key] = value
    try:
        model.load_state_dict(new_state_dict)
    except Exception as e:
        print(f"Error loading weights: {e}")
        return
    model.eval()
    tta_preprocess = transforms.Compose([
        transforms.Resize(256),       
        transforms.FiveCrop(224),     
    ])
    norm_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    stats = {code: {'1st': 0, '2nd': 0, '3rd': 0, 'miss': 0, 'total': 0} for code in PREFIX_MAP.keys()}
    continent_stats = {} 
    valid_files = [f for f in os.listdir(TEST_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(valid_files)} images in {TEST_FOLDER}. Running inference...")
    for fname in valid_files:
        prefix = fname[:2].lower()
        if prefix not in PREFIX_MAP: continue
        true_name = PREFIX_MAP[prefix]
        true_continent = CITY_TO_CONTINENT.get(true_name, "Unknown")
        if true_continent not in continent_stats:
            continent_stats[true_continent] = {'1st': 0, '2nd': 0, '3rd': 0, 'miss': 0, 'total': 0}
        img_path = os.path.join(TEST_FOLDER, fname)
        try:
            image = Image.open(img_path).convert("RGB")
            crops = tta_preprocess(image)
            input_tensor = torch.stack([norm_transform(c) for c in crops]).to(DEVICE)
            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.nn.functional.softmax(logits, dim=1)
                avg_prob = probs.mean(dim=0)
            top3_prob, top3_idx = torch.topk(avg_prob, 3)
            preds = [classes[i] for i in top3_idx]
            stats[prefix]['total'] += 1
            if true_name == preds[0]:
                stats[prefix]['1st'] += 1
            elif true_name == preds[1]:
                stats[prefix]['2nd'] += 1
                print(f"⚠️ {fname} -> {preds[0]} (True: {true_name} 2nd)")
            elif true_name == preds[2]:
                stats[prefix]['3rd'] += 1
                print(f"⚠️ {fname} -> {preds[0]} (True: {true_name} 3rd)")
            else:
                stats[prefix]['miss'] += 1
                print(f"❌ {fname} -> {preds[0]} (True: {true_name} missed)")
            continent_stats[true_continent]['total'] += 1
            pred_cont_1 = CITY_TO_CONTINENT.get(preds[0], "Unknown")
            pred_cont_2 = CITY_TO_CONTINENT.get(preds[1], "Unknown")
            pred_cont_3 = CITY_TO_CONTINENT.get(preds[2], "Unknown")
            if pred_cont_1 == true_continent:
                continent_stats[true_continent]['1st'] += 1
            elif pred_cont_2 == true_continent:
                continent_stats[true_continent]['2nd'] += 1
            elif pred_cont_3 == true_continent:
                continent_stats[true_continent]['3rd'] += 1
            else:
                continent_stats[true_continent]['miss'] += 1
        except Exception as e:
            print(f"Err {fname}: {e}")
    print("\n" + "="*60)
    print(f"{'CITY':<15} | {'TOTAL':<5} | {'1st (Correct)':<13} | {'2nd':<5} | {'3rd':<5} | {'MISS':<5}")
    print("-" * 60)
    total_hits = 0
    total_imgs = 0
    for p, d in stats.items():
        if d['total'] > 0:
            name = PREFIX_MAP[p].upper()
            acc = 100 * d['1st'] / d['total']
            print(f"{name:<15} | {d['total']:<5} | {d['1st']} ({acc:.0f}%)      | {d['2nd']:<5} | {d['3rd']:<5} | {d['miss']:<5}")
            total_hits += d['1st']
            total_imgs += d['total']
    if total_imgs > 0:
        print("-" * 60)
        print(f"OVERALL ACCURACY: {total_hits}/{total_imgs} = {100*total_hits/total_imgs:.1f}%")
    print("\n" + "="*60)
    print(f"{'CONTINENT':<15} | {'TOTAL':<5} | {'1st (Correct)':<13} | {'2nd':<5} | {'3rd':<5} | {'MISS':<5}")
    print("-" * 60)
    for cont, data in continent_stats.items():
        t = data['total']
        if t > 0:
            p1 = f"{data['1st']} ({int(data['1st']/t*100)}%)"
            print(f"{cont:<15} | {t:<5} | {p1:<13} | {data['2nd']:<5} | {data['3rd']:<5} | {data['miss']:<5}")
    print("="*60)
if __name__ == "__main__":
    main()