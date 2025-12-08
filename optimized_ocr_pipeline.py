import os
import torch
import numpy as np
import easyocr
import re
import cv2
import pickle
import glob
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader

# --- CONFIGURAÇÃO ---
DATASET_ROOT = "data"
OUTPUT_FILE = "ocr_features_smart.pkl"
CHECKPOINT_FREQ = 2000
IMG_SIZE_OCR = 640
BATCH_SIZE_EMBEDDING = 64
NUM_WORKERS = 8 
DEVICE = "cuda"

# --- MAPEAMENTO DE CIDADES (LÓGICA DO USUÁRIO) ---
# Define qual perfil de linguagem cada pasta de cidade usa.
# Se a cidade não estiver aqui, usa o padrão 'latin' (apenas inglês)
CITY_PROFILES = {
    'bangkok': 'thai',
    'tokyo': 'japanese',
    'moscow': 'cyrillic',
    'goa': 'konkani',
    'bengaluru': 'kannada',
    'athens': 'greek' 
}

# --- FUNÇÕES AUXILIARES ---
def clean_text(text):
    text = re.sub(r'm[ao]p[il]+ary', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+[bcdefghjklmnopqrstuvwxyz]\s+', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def detect_scripts(text):
    # Detecta proporção de scripts para vetor de features
    counts = {
        'latin': len(re.findall(r'[a-zA-Z]', text)),
        'cyrillic': len(re.findall(r'[\u0400-\u04FF]', text)), 
        'greek': len(re.findall(r'[\u0370-\u03FF]', text)),    
        'thai': len(re.findall(r'[\u0E00-\u0E7F]', text)),     
        'japanese': len(re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', text)),
        'indic': len(re.findall(r'[\u0900-\u097F\u0C80-\u0CFF]', text))
    }
    total = sum(counts.values())
    vec = np.zeros(6, dtype=np.float32)
    if total > 0:
        vec[0] = counts['latin'] / total
        vec[1] = counts['cyrillic'] / total
        vec[2] = counts['greek'] / total
        vec[3] = counts['thai'] / total
        vec[4] = counts['japanese'] / total
        vec[5] = counts['indic'] / total
    return vec, total > 0

# --- DATASET ---
class SmartOCRDataset(Dataset):
    def __init__(self, root_dir):
        self.items = [] # Tuplas (path, city_key)
        print("📂 Scanning directory structure...")
        
        for city in tqdm(os.listdir(root_dir), desc="Scanning Cities"):
            city_path = os.path.join(root_dir, city)
            if not os.path.isdir(city_path): continue
            
            # Determina o perfil da cidade
            # Se a pasta for 'bangkok', usa 'thai', senão 'latin'
            profile_key = CITY_PROFILES.get(city.lower(), 'latin')
            
            for sub in ['database', 'query']:
                img_dir = os.path.join(city_path, sub, 'images')
                if not os.path.exists(img_dir): continue
                
                files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                for f in files:
                    full_path = os.path.join(img_dir, f)
                    self.items.append((full_path, profile_key))
        
        # Sort para garantir determinismo
        self.items.sort(key=lambda x: x[0])
        print(f"✅ Found {len(self.items)} images.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, profile_key = self.items[idx]
        
        try:
            img = cv2.imread(path)
            if img is None: raise ValueError("Img None")
            
            h, w = img.shape[:2]
            scale = IMG_SIZE_OCR / max(h, w)
            if scale < 1.0:
                new_w, new_h = int(w * scale), int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            return img, path, profile_key, True
        except:
            return np.zeros((10, 10, 3), dtype=np.uint8), path, "latin", False

def custom_collate(batch):
    # Desempacota o batch
    return [x[0] for x in batch], [x[1] for x in batch], [x[2] for x in batch], [x[3] for x in batch]

# --- MAIN ---
def main():
    print("🚀 Initializing SMART OCR Pipeline (City-Aware)...")
    
    # 1. Carregar Checkpoint
    data_cache = {}
    if os.path.exists(OUTPUT_FILE):
        print(f"🔄 Resuming from {OUTPUT_FILE}...")
        with open(OUTPUT_FILE, 'rb') as f:
            data_cache = pickle.load(f)
        print(f"   -> {len(data_cache)} images already processed.")
    
    dataset = SmartOCRDataset(DATASET_ROOT)
    
    # Filtro de já processados
    # Criamos um dict temporário só com caminhos para filtrar rápido
    processed_paths = set(data_cache.keys())
    remaining_indices = [i for i, (p, _) in enumerate(dataset.items) if p not in processed_paths]
    
    if not remaining_indices:
        print("🎉 All images already processed!")
        return

    # Reduz dataset apenas para o que falta
    dataset.items = [dataset.items[i] for i in remaining_indices]
    print(f"📉 Processing remaining {len(dataset)} images...")

    loader = DataLoader(
        dataset, 
        batch_size=1,     
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        collate_fn=custom_collate
    )

    # 2. Carregar Modelos em um Dicionário
    print("🧠 Loading Specialist Models to GPU...")
    
    # Dicionário de Readers
    readers_bank = {}
    
    configs = {
        'latin':    ['en'],           # Padrão (Rápido)
        'thai':     ['th', 'en'],     # Bangkok
        'japanese': ['ja', 'en'],     # Tokyo
        'cyrillic': ['ru', 'en'],     # Moscow
        'konkani':  ['gom', 'en'],    # Goa
        'kannada':  ['kn', 'en'],     # Bengaluru
    }

    for key, langs in configs.items():
        print(f"  -> Loading {key} reader {langs}...")
        try:
            readers_bank[key] = easyocr.Reader(langs, gpu=True, quantize=False, verbose=False)
        except Exception as e:
            print(f"  ⚠️ Failed to load {key}: {e}. Fallback to Latin.")
            # Se falhar (ex: grego), aponta para o leitor latino existente (se houver) ou cria um
            if 'latin' in readers_bank:
                readers_bank[key] = readers_bank['latin']
            else:
                readers_bank['latin'] = easyocr.Reader(['en'], gpu=True)
                readers_bank[key] = readers_bank['latin']

    embedder = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
    print("✅ Models Loaded.")

    # Buffers
    text_buffer = []
    path_buffer = []
    script_buffer = []

    # Loop Principal
    for images, paths, profiles, valids in tqdm(loader, desc="Smart OCR Processing"):
        img = images[0]
        path = paths[0]
        profile_key = profiles[0] # Ex: 'thai', 'latin'
        is_valid = valids[0]

        if not is_valid:
            data_cache[path] = (np.zeros(6, dtype=np.float32), np.zeros(384, dtype=np.float32))
            continue

        full_text = ""
        unique_words = set()
        script_vec = np.zeros(6, dtype=np.float32)
        
        try:
            # --- MÁGICA AQUI ---
            # Seleciona APENAS o leitor correto para esta cidade
            reader = readers_bank.get(profile_key, readers_bank.get('latin'))
            
            res = reader.readtext(img, detail=0, paragraph=True)
            for seg in res:
                c = clean_text(seg)
                if c and c not in unique_words:
                    unique_words.add(c)
                    full_text += " " + c
            
            full_text = full_text.strip()
            script_vec, has_text = detect_scripts(full_text)
            if not has_text: full_text = ""

        except Exception:
            full_text = ""
        
        # Bufferização (Mantida igual para eficiência do Transformer)
        text_buffer.append(full_text if full_text else "empty")
        path_buffer.append(path)
        script_buffer.append(script_vec)

        if len(text_buffer) >= BATCH_SIZE_EMBEDDING:
            embeddings = embedder.encode(text_buffer, batch_size=BATCH_SIZE_EMBEDDING, show_progress_bar=False, convert_to_numpy=True)
            
            for p, s, e, t in zip(path_buffer, script_buffer, embeddings, text_buffer):
                if t == "empty": e = np.zeros_like(e)
                data_cache[p] = (s, e)
            
            text_buffer, path_buffer, script_buffer = [], [], []

            if len(data_cache) % CHECKPOINT_FREQ < BATCH_SIZE_EMBEDDING:
                with open(OUTPUT_FILE, 'wb') as f: pickle.dump(data_cache, f)

    # Final Flush
    if text_buffer:
        embeddings = embedder.encode(text_buffer, batch_size=len(text_buffer), show_progress_bar=False, convert_to_numpy=True)
        for p, s, e, t in zip(path_buffer, script_buffer, embeddings, text_buffer):
            if t == "empty": e = np.zeros_like(e)
            data_cache[p] = (s, e)

    with open(OUTPUT_FILE, 'wb') as f: pickle.dump(data_cache, f)
    print(f"🎉 Done! Saved {len(data_cache)} features.")

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()