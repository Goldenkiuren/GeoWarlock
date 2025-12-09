import os
import torch
import torch.nn as nn
import pandas as pd
import argparse
from PIL import Image
from torchvision import transforms, models
import numpy as np
import easyocr
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# --- CONFIGURAÇÃO ---
TEST_FOLDER = "test"
DATASET_ROOT = "data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IGNORED_CITIES = ['amman', 'nairobi']

# Mapeamentos (Mesmos do seu arquivo original)
PREFIX_MAP = {
    'sp': 'saopaulo', 'ba': 'buenosaires', 'am': 'amsterdam', 'bk': 'bangkok',
    'he': 'helsinki', 'kp': 'kampala', 'me': 'melbourne', 'mi': 'miami',
    'sf': 'sf', 'tk': 'tokyo', 'go': 'goa', 'ma': 'manila', 'at': 'athens',
    'pa': 'paris', 'bu': 'budapest', 'ot': 'ottawa', 'au': 'austin',
    'be': 'bengaluru', 'br': 'berlin', 'bo': 'boston', 'co': 'cph', # Atenção: copenhagen virou cph
    'lo': 'london', 'mo': 'moscow', 'ph': 'phoenix', 'st': 'stockholm',
    'tr': 'trondheim', 'to': 'toronto', 'zu': 'zurich'
}

CITY_TO_CONTINENT = {
    'saopaulo': 'South America', 'buenosaires': 'South America',
    'miami': 'North America', 'sf': 'North America', 'ottawa': 'North America',
    'austin': 'North America', 'boston': 'North America', 'phoenix': 'North America',
    'toronto': 'North America', 'amsterdam': 'Europe', 'helsinki': 'Europe',
    'athens': 'Europe', 'paris': 'Europe', 'budapest': 'Europe', 'berlin': 'Europe',
    'cph': 'Europe', 'london': 'Europe', 'moscow': 'Europe',
    'stockholm': 'Europe', 'trondheim': 'Europe', 'zurich': 'Europe',
    'bangkok': 'Asia', 'tokyo': 'Asia', 'goa': 'Asia', 'manila': 'Asia',
    'bengaluru': 'Asia', 'kampala': 'Africa', 'melbourne': 'Oceania'
}

# --- MODELO (Cópia Exata) ---
class MultiModalViT(nn.Module):
    def __init__(self, num_classes):
        super(MultiModalViT, self).__init__()
        self.vit = models.vit_b_16(weights=None)
        visual_dim = self.vit.heads.head.in_features
        self.vit.heads = nn.Identity() 
        self.text_process = nn.Sequential(
            nn.Linear(390, 512), 
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
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

# --- HELPER DE OCR (COM BLACKLIST ANTI-GOOGLE & MULTI-SCRIPT) ---
class LiveOCRProcessor:
    def __init__(self):
        self.readers = []
        
        # Configuração de Leitores (Estratégia Ensemble)
        configs = [
            ('thai', ['th', 'en']),
            ('chinese', ['ch_sim', 'en']),
            ('japanese', ['ja', 'en']),
            ('korean', ['ko', 'en']),
            ('cyrillic', ['ru', 'en']),
            ('latin', ['pt', 'es', 'en']) # Português/Espanhol/Inglês para ocidente
        ]
        
        print("Loading OCR Engines...")
        for name, langs in configs:
            try:
                # Carrega silenciosamente
                self.readers.append((name, easyocr.Reader(langs, gpu=True, verbose=False)))
            except Exception as e:
                print(f"Warning: Could not load {name}: {e}")

        self.sent_model = SentenceTransformer('all-MiniLM-L6-v2')

    def detect_script(self, text):
        counts = np.zeros(6, dtype=np.float32)
        if not text: return counts
        for char in text:
            cp = ord(char)
            if 0x0041 <= cp <= 0x007A: counts[0] += 1  # Latin
            elif 0x0400 <= cp <= 0x04FF: counts[1] += 1  # Cyrillic
            elif 0x0E00 <= cp <= 0x0E7F: counts[2] += 1  # Thai
            elif 0x4E00 <= cp <= 0x9FFF: counts[3] += 1  # Chinese
            elif 0x3040 <= cp <= 0x30FF: counts[4] += 1  # Japanese
            elif 0xAC00 <= cp <= 0xD7AF: counts[5] += 1  # Korean
        total = counts.sum()
        if total > 0: counts /= total
        return counts

    def process(self, image_path):
        unique_text = set()
        
# --- LISTA NEGRA AGRESSIVA (ANTI-UI & ANTI-RUÍDO) ---
        BLACKLIST = [
            # Variações do Google (O OCR erra muito aqui)
            "google", "gogle", "googie", "gooole", "gocgle", "goco", "gooulo", 
            "gool", "goggle", "gol", "oogle", "ogle", "gugle", "geo", "2019", "202",
            
            # Interface PT/EN
            "maps", "map", "pesquise", "search", "street", "view", "stree",
            "capture", "captura", "copyright", "copy", "©", "termos", "terms",
            "privacidade", "privacy", "report", "problem", "problema", "atale",
            "data", "date", "imagem", "image", "bairro", "r.", "rua", 
            "compartilhar", "share", "ver mais", "view more", "ano",
            
            # Ruídos comuns de OCR asiático em imagens ocidentais
            "ooale", "อี้", "tcp", "027", "711", "...", "::", "///"
        ]

        try:
            for name, reader in self.readers:
                try:
                    # Parâmetros otimizados para leitura difícil
                    results = reader.readtext(image_path, detail=1, adjust_contrast=0.5, mag_ratio=1.5, paragraph=False)
                    
                    for (_, text, prob) in results:
                        # 1. Filtro de Confiança (Mais rigoroso para asiáticos)
                        min_conf = 0.3 if name == 'latin' else 0.5
                        
                        if prob > min_conf:
                            clean = text.strip()
                            lower_clean = clean.lower()
                            
                            # 2. Filtro de Tamanho (Ignora lixo curto)
                            if len(clean) <= 2:
                                continue
                                
                            # 3. FILTRO ANTI-UI (Remove interface do Google)
                            if any(bad_word in lower_clean for bad_word in BLACKLIST):
                                continue
                                
                            unique_text.add(clean)
                except: continue

            full_text = " ".join(unique_text)
            
            # Embeddings
            if not full_text.strip():
                emb = np.zeros(384, dtype=np.float32)
            else:
                emb = self.sent_model.encode([full_text])[0]
            
            script = self.detect_script(full_text)
            combined = np.concatenate([script, emb], axis=0)
            
            # Retorna Tensor e Texto (para debug se necessário)
            # Nota: para os scripts batch, retornamos apenas o tensor na chamada principal, 
            # mas mantemos a assinatura compatível retornando tupla ou ajustando no script.
            # PARA O BATCH SCRIPT, O ESPERADO É APENAS O TENSOR.
            return torch.tensor(combined, dtype=torch.float32).to(DEVICE)
            
        except:
            return torch.zeros(390, dtype=torch.float32).to(DEVICE)

def process(self, image_path):
        unique_text = set()
        
        # --- LISTA NEGRA DE UI (Interface do Google) ---
        # Adicione aqui qualquer texto de interface que aparecer nos logs
        BLACKLIST = [
            "google", "maps", "pesquise", "search", "street view", 
            "image capture", "copyright", "©", "202", "termos", 
            "privacidade", "report", "problem", "atale"
        ]

        try:
            for name, reader in self.readers:
                try:
                    # Leitura
                    results = reader.readtext(image_path, detail=1, adjust_contrast=0.5, mag_ratio=1.5, paragraph=False)
                    
                    for (_, text, prob) in results:
                        # 1. Filtro de Confiança
                        min_conf = 0.3 if name == 'latin' else 0.5
                        
                        if prob > min_conf:
                            clean = text.strip()
                            lower_clean = clean.lower()
                            
                            # 2. Filtro de Tamanho (Ruído curto)
                            if len(clean) <= 2:
                                continue
                                
                            # 3. FILTRO DE BLACKLIST (Novo!)
                            # Se qualquer palavra proibida estiver no texto, descarta.
                            if any(bad_word in lower_clean for bad_word in BLACKLIST):
                                continue
                                
                            unique_text.add(clean)
                except: continue

            full_text = " ".join(unique_text)
            
            # Embeddings
            if not full_text.strip():
                emb = np.zeros(384, dtype=np.float32)
            else:
                emb = self.sent_model.encode([full_text])[0]
            
            script = self.detect_script(full_text)
            combined = np.concatenate([script, emb], axis=0)
            return torch.tensor(combined, dtype=torch.float32).to(DEVICE), full_text # Retornando full_text para debug
            
        except:
            return torch.zeros(390, dtype=torch.float32).to(DEVICE), ""

def get_classes(dataset_root):
    classes = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
    classes.sort()
    classes = [c for c in classes if c not in IGNORED_CITIES]
    return classes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="best_multimodal_model.pth")
    args = parser.parse_args()

    # 1. Setup
    classes = get_classes(DATASET_ROOT)
    num_classes = len(classes)
    print(f"Loaded {num_classes} classes.")
    
    # 2. Carregar OCR e Modelo
    print("Initializing OCR Engine (this takes a moment)...")
    ocr = LiveOCRProcessor()
    
    print(f"Loading Model: {args.model}")
    model = MultiModalViT(num_classes).to(DEVICE)
    checkpoint = torch.load(args.model, map_location=DEVICE)
    new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}
    model.load_state_dict(new_state_dict)
    model.eval()

    # 3. Transformações (FiveCrop para TTA)
    tta_preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.FiveCrop(224),
    ])
    norm_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # 4. Loop de Teste
    stats = {code: {'1st': 0, '2nd': 0, '3rd': 0, 'miss': 0, 'total': 0} for code in PREFIX_MAP.keys()}
    continent_stats = {}
    
    valid_files = [f for f in os.listdir(TEST_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(valid_files)} images. Running Multimodal Inference...")

    for fname in tqdm(valid_files):
        prefix = fname[:2].lower()
        if prefix not in PREFIX_MAP: continue
        
        true_class = PREFIX_MAP[prefix]
        true_cont = CITY_TO_CONTINENT.get(true_class, "Unknown")
        if true_cont not in continent_stats:
            continent_stats[true_cont] = {'1st': 0, '2nd': 0, '3rd': 0, 'miss': 0, 'total': 0}

        path = os.path.join(TEST_FOLDER, fname)
        
        try:
            # A. Processamento OCR (gera vetor de 390)
            txt_tensor_single = ocr.process(path) 
            
            # B. Processamento Visual (gera 5 crops)
            image = Image.open(path).convert("RGB")
            crops = tta_preprocess(image)
            img_batch = torch.stack([norm_transform(crop) for crop in crops]).to(DEVICE)
            
            # C. Replicar Texto para TTA (batch size 5)
            # O texto é o mesmo para todos os 5 crops da mesma imagem
            txt_batch = txt_tensor_single.unsqueeze(0).repeat(5, 1)

            # D. Inferência
            with torch.no_grad():
                outputs = model(img_batch, txt_batch)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                avg_probs = torch.mean(probs, dim=0, keepdim=True)
                top3_prob, top3_idx = torch.topk(avg_probs, 3)
            
            top3_names = [classes[idx] for idx in top3_idx[0]]
            
            # Atualizar Stats
            stats[prefix]['total'] += 1
            if true_class == top3_names[0]: stats[prefix]['1st'] += 1
            elif true_class == top3_names[1]: stats[prefix]['2nd'] += 1
            elif true_class == top3_names[2]: stats[prefix]['3rd'] += 1
            else: stats[prefix]['miss'] += 1

            # Atualizar Continent Stats
            continent_stats[true_cont]['total'] += 1
            pred_conts = [CITY_TO_CONTINENT.get(n, "U") for n in top3_names]
            if pred_conts[0] == true_cont: continent_stats[true_cont]['1st'] += 1
            elif pred_conts[1] == true_cont: continent_stats[true_cont]['2nd'] += 1
            elif pred_conts[2] == true_cont: continent_stats[true_cont]['3rd'] += 1
            else: continent_stats[true_cont]['miss'] += 1

        except Exception as e:
            print(f"Error on {fname}: {e}")

    # --- RELATÓRIO ---
    print("\n" + "="*60)
    print(f"{'CITY':<15} | {'TOT':<5} | {'1st (Correct)':<13} | {'2nd':<5} | {'3rd':<5} | {'MISS':<5}")
    print("-" * 60)
    
    total_correct = 0
    total_imgs = 0
    for p, d in stats.items():
        if d['total'] == 0: continue
        t = d['total']
        print(f"{PREFIX_MAP[p].upper():<15} | {t:<5} | {d['1st']} ({int(d['1st']/t*100)}%)    | {d['2nd']:<5} | {d['3rd']:<5} | {d['miss']:<5}")
        total_correct += d['1st']
        total_imgs += t
        
    print("-" * 60)
    if total_imgs > 0:
        print(f"OVERALL ACCURACY: {total_correct/total_imgs*100:.1f}%")
    print("="*60)

if __name__ == "__main__":
    main()