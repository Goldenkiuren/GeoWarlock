import os
import torch
import pandas as pd
import argparse  
from PIL import Image
from torchvision import transforms, models
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
def main():
    parser = argparse.ArgumentParser(description="Run City Classification Inference")
    parser.add_argument("--model", type=str, default="best_models/vit16_best.pth", 
                        help="Path to the .pth model file to use")
    args = parser.parse_args()
    model_path = args.model
    classes = get_classes(DATASET_ROOT)
    num_classes = len(classes)
    print(f"Loaded {num_classes} classes (Validation Mode).")
    model = ViTForCityClassification(num_classes=num_classes).to(DEVICE)
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        return
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=DEVICE)
    new_state_dict = {}
    for key, value in checkpoint.items():
        if key.startswith("_orig_mod."):
            new_key = key.replace("_orig_mod.", "")
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    model.load_state_dict(new_state_dict)
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
    print(f"\nScanning folder '{TEST_FOLDER}'...")
    valid_files = [f for f in os.listdir(TEST_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not valid_files:
        print("No images found! Check the folder name.")
        return
    print(f"Found {len(valid_files)} images. Running inference with TTA (FiveCrop)...\n")
    for fname in valid_files:
        prefix = fname[:2].lower()
        if prefix not in PREFIX_MAP:
            print(f"Skipping {fname}: Unknown prefix '{prefix}'")
            continue
        true_class_name = PREFIX_MAP[prefix]
        true_continent = CITY_TO_CONTINENT.get(true_class_name, "Unknown")
        if true_continent not in continent_stats:
            continent_stats[true_continent] = {'1st': 0, '2nd': 0, '3rd': 0, 'miss': 0, 'total': 0}
        img_path = os.path.join(TEST_FOLDER, fname)
        try:
            image = Image.open(img_path).convert("RGB")
            crops = tta_preprocess(image)
            input_tensor = torch.stack([norm_transform(crop) for crop in crops]).to(DEVICE)
        except Exception as e:
            print(f"Error reading {fname}: {e}")
            continue
        with torch.no_grad():
            outputs = model(input_tensor) 
            probs_per_crop = torch.nn.functional.softmax(outputs, dim=1)
            avg_probs = torch.mean(probs_per_crop, dim=0, keepdim=True) 
            top3_prob, top3_idx = torch.topk(avg_probs, 3)
        top3_names = [classes[idx] for idx in top3_idx[0]]
        stats[prefix]['total'] += 1
        if true_class_name == top3_names[0]:
            stats[prefix]['1st'] += 1
        elif true_class_name == top3_names[1]:
            stats[prefix]['2nd'] += 1
            print(f"⚠️ {fname} -> {top3_names[0]} (True: {true_class_name} was 2nd)")
        elif true_class_name == top3_names[2]:
            stats[prefix]['3rd'] += 1
            print(f"⚠️ {fname} -> {top3_names[0]} (True: {true_class_name} was 3rd)")
        else:
            stats[prefix]['miss'] += 1
            print(f"❌ {fname} -> {top3_names[0]} (True: {true_class_name} not in top 3)")
        continent_stats[true_continent]['total'] += 1
        pred_cont_1 = CITY_TO_CONTINENT.get(top3_names[0], "Unknown")
        pred_cont_2 = CITY_TO_CONTINENT.get(top3_names[1], "Unknown")
        pred_cont_3 = CITY_TO_CONTINENT.get(top3_names[2], "Unknown")
        if pred_cont_1 == true_continent:
            continent_stats[true_continent]['1st'] += 1
        elif pred_cont_2 == true_continent:
            continent_stats[true_continent]['2nd'] += 1
        elif pred_cont_3 == true_continent:
            continent_stats[true_continent]['3rd'] += 1
        else:
            continent_stats[true_continent]['miss'] += 1
    print("\n" + "="*60)
    print(f"{'CITY':<15} | {'TOTAL':<5} | {'1st (Correct)':<13} | {'2nd':<5} | {'3rd':<5} | {'MISS':<5}")
    print("-" * 60)
    total_correct = 0
    total_imgs = 0
    for prefix, data in stats.items():
        if data['total'] == 0: continue
        city_name = PREFIX_MAP[prefix].upper()
        t = data['total']
        p1 = f"{data['1st']} ({int(data['1st']/t*100)}%)"
        print(f"{city_name:<15} | {t:<5} | {p1:<13} | {data['2nd']:<5} | {data['3rd']:<5} | {data['miss']:<5}")
        total_correct += data['1st']
        total_imgs += t
    print("-" * 60)
    if total_imgs > 0:
        print(f"OVERALL ACCURACY: {total_correct}/{total_imgs} = {total_correct/total_imgs*100:.1f}%")
    print("\n" + "="*60)
    print(f"{'CONTINENT':<15} | {'TOTAL':<5} | {'1st (Correct)':<13} | {'2nd':<5} | {'3rd':<5} | {'MISS':<5}")
    print("-" * 60)
    for cont, data in continent_stats.items():
        t = data['total']
        p1 = f"{data['1st']} ({int(data['1st']/t*100)}%)"
        print(f"{cont:<15} | {t:<5} | {p1:<13} | {data['2nd']:<5} | {data['3rd']:<5} | {data['miss']:<5}")
    print("="*60)
if __name__ == "__main__":
    main()