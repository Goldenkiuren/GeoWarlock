import os
import torch
import torch.nn as nn
import numpy as np
import easyocr
from sentence_transformers import SentenceTransformer
from PIL import Image
from torchvision import transforms, models
from tqdm import tqdm

# --- CONFIGURAÇÃO ---
TEST_FOLDER = "test"
DATASET_ROOT = "data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH_RANGE = range(1, 16)  # Testa de 8 até 15

# Mapeamento de Prefixos
PREFIX_MAP = {
    'sp': 'saopaulo', 'ba': 'buenosaires', 'am': 'amsterdam', 'bk': 'bangkok',
    'he': 'helsinki', 'kp': 'kampala', 'me': 'melbourne', 'mi': 'miami',
    'sf': 'sf', 'tk': 'tokyo', 'go': 'goa', 'ma': 'manila', 'at': 'athens',
    'pa': 'paris', 'bu': 'budapest', 'ot': 'ottawa', 'au': 'austin',
    'be': 'bengaluru', 'br': 'berlin', 'bo': 'boston', 'co': 'cph',
    'lo': 'london', 'mo': 'moscow', 'ph': 'phoenix', 'st': 'stockholm',
    'tr': 'trondheim', 'to': 'toronto', 'zu': 'zurich'
}
IGNORED_CITIES = ['amman', 'nairobi']

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
        
        # --- A LISTA NEGRA (O SEGREDO) ---
        # Qualquer texto que contenha estas palavras será ignorado
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

# --- MODELO ---
class MultiModalViT(nn.Module):
    def __init__(self, num_classes):
        super(MultiModalViT, self).__init__()
        self.vit = models.vit_b_16(weights=None)
        self.vit.heads = nn.Identity() 
        self.text_process = nn.Sequential(
            nn.Linear(390, 512), 
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3)
        )
        self.classifier = nn.Sequential(
            nn.Linear(768 + 512, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, num_classes)
        )

    def forward(self, img, text_feat):
        vis_feat = self.vit(img)
        txt_feat_proc = self.text_process(text_feat)
        combined = torch.cat((vis_feat, txt_feat_proc), dim=1)
        return self.classifier(combined)

def get_classes(dataset_root):
    classes = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
    classes.sort()
    return [c for c in classes if c not in IGNORED_CITIES]

def main():
    # 1. Preparação
    classes = get_classes(DATASET_ROOT)
    num_classes = len(classes)
    print(f"Classes: {num_classes}")
    
    ocr = LiveOCRProcessor()
    
    valid_files = [f for f in os.listdir(TEST_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not valid_files: return print("No images found.")

    # 2. CACHE DE OCR (A Grande Otimização)
    # Processamos todas as imagens uma única vez e guardamos na RAM
    print(f"\n--- STEP 1: Pre-computing OCR for {len(valid_files)} images ---")
    ocr_cache = {}
    
    for fname in tqdm(valid_files, desc="OCR Processing"):
        path = os.path.join(TEST_FOLDER, fname)
        ocr_cache[fname] = ocr.process(path)
    
    # 3. SETUP DAS IMAGENS VISUAIS (TTA)
    tta_preprocess = transforms.Compose([
        transforms.Resize(256), transforms.FiveCrop(224),
    ])
    norm_transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # 4. LOOP PELAS ÉPOCAS
    results_summary = []

    print("\n--- STEP 2: Testing Models ---")
    
    for epoch in EPOCH_RANGE:
        model_name = f"multimodal_model_epoch_{epoch}.pth"
        if not os.path.exists(model_name):
            print(f"Skipping {model_name} (Not found)")
            continue
            
        print(f"\nTesting Epoch {epoch}...")
        
        # Carregar Modelo
        model = MultiModalViT(num_classes).to(DEVICE)
        checkpoint = torch.load(model_name, map_location=DEVICE, weights_only=True)
        new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}
        model.load_state_dict(new_state_dict)
        model.eval()
        
        correct = 0
        total = 0
        
        # Inferência Rápida (OCR já está pronto)
        for fname in tqdm(valid_files, desc=f"Ep {epoch} Inference", leave=False):
            prefix = fname[:2].lower()
            if prefix not in PREFIX_MAP: continue
            true_class = PREFIX_MAP[prefix]
            
            # Recupera OCR do Cache
            txt_tensor_single = ocr_cache[fname]
            
            # Prepara Imagem TTA
            path = os.path.join(TEST_FOLDER, fname)
            try:
                img = Image.open(path).convert("RGB")
                crops = tta_preprocess(img)
                img_batch = torch.stack([norm_transform(crop) for crop in crops]).to(DEVICE)
                
                # Expande Texto para batch 5
                txt_batch = txt_tensor_single.unsqueeze(0).repeat(5, 1)
                
                with torch.no_grad():
                    outputs = model(img_batch, txt_batch)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    avg_probs = torch.mean(probs, dim=0, keepdim=True)
                    _, top1_idx = torch.max(avg_probs, 1)
                
                pred_class = classes[top1_idx.item()]
                
                if pred_class == true_class:
                    correct += 1
                total += 1
                
            except Exception as e:
                print(f"Err {fname}: {e}")

        acc = (correct / total * 100) if total > 0 else 0
        print(f"  -> Epoch {epoch} Accuracy: {acc:.2f}% ({correct}/{total})")
        results_summary.append((epoch, acc))

    # 5. TABELA FINAL
    print("\n" + "="*30)
    print("   FINAL RESULTS SUMMARY")
    print("="*30)
    print(f"{'EPOCH':<10} | {'ACCURACY':<10}")
    print("-" * 30)
    
    best_ep = -1
    best_acc = -1
    
    for ep, acc in results_summary:
        print(f"{ep:<10} | {acc:.2f}%")
        if acc > best_acc:
            best_acc = acc
            best_ep = ep
            
    print("-" * 30)
    print(f"🏆 WINNER: Epoch {best_ep} ({best_acc:.2f}%)")
    print("="*30)

if __name__ == "__main__":
    main()