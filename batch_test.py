import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn

# --- CONFIGURAÇÃO ---
TEST_FOLDER = "test"        # Nome da pasta onde estão suas imagens
DATASET_ROOT = "data"       # Pasta original para ler os nomes das classes
MODEL_PATH = "best_city_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mapeamento do prefixo do arquivo para o nome EXATO da pasta da classe
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
    'tk': 'tokyo'
}

# Cidades que o treino ignorou
IGNORED_CITIES = ['amman', 'nairobi']

def get_classes(dataset_root):
    classes = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
    classes.sort()
    # Remove as cidades que foram puladas no treino
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
    # 1. Carregar Classes
    classes = get_classes(DATASET_ROOT)
    num_classes = len(classes)
    print(f"Loaded {num_classes} classes (Validation Mode).")

    # 2. Carregar Modelo
    model = ViTForCityClassification(num_classes=num_classes).to(DEVICE)
    
    # --- CORREÇÃO DO TORCH.COMPILE (Fix do _orig_mod) ---
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
    
    # 3. Definir Transformação (A parte que estava faltando)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 4. Preparar Estatísticas e Rodar
    stats = {code: {'1st': 0, '2nd': 0, '3rd': 0, 'miss': 0, 'total': 0} for code in PREFIX_MAP.keys()}
    
    print(f"\nScanning folder '{TEST_FOLDER}'...")
    valid_files = [f for f in os.listdir(TEST_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not valid_files:
        print("No images found! Check the folder name.")
        return

    print(f"Found {len(valid_files)} images. Running inference...\n")

    for fname in valid_files:
        prefix = fname[:2].lower()
        
        if prefix not in PREFIX_MAP:
            print(f"Skipping {fname}: Unknown prefix '{prefix}'")
            continue
            
        true_class_name = PREFIX_MAP[prefix]
        
        img_path = os.path.join(TEST_FOLDER, fname)
        try:
            image = Image.open(img_path).convert("RGB")
            # Agora 'transform' está definido!
            input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        except Exception as e:
            print(f"Error reading {fname}: {e}")
            continue

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            top3_prob, top3_idx = torch.topk(probs, 3)
            
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
    print("="*60)

if __name__ == "__main__":
    main()