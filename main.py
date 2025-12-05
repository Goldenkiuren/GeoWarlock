import os
import pandas as pd
import random
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# --- DATASET (Mantido igual, apenas compactado aqui para contexto) ---
class MSLSTripletDataset(Dataset):
    def __init__(self, root_dir, split='train_val', transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.sequence_map = {} 
        self.valid_keys = []
        
        if not os.path.exists(self.root_dir):
            raise ValueError(f"Diretório não encontrado: {self.root_dir}")
            
        cities = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        print(f"Indexando sequências de {len(cities)} cidades...")
        
        for city in cities:
            db_path = os.path.join(self.root_dir, city, 'database')
            images_path = os.path.join(db_path, 'images')
            csv_path = os.path.join(db_path, 'seq_info.csv') # Usamos seq_info, não subtask_index
            
            if os.path.exists(csv_path) and os.path.exists(images_path):
                df = pd.read_csv(csv_path, usecols=['key', 'sequence_key'])
                for _, row in df.iterrows():
                    seq_key = row['sequence_key']
                    img_key = row['key']
                    full_img_path = os.path.join(images_path, f"{img_key}.jpg")
                    
                    # Verificação rápida de existência (opcional se confiar nos dados)
                    # if os.path.exists(full_img_path): 
                    if seq_key not in self.sequence_map:
                        self.sequence_map[seq_key] = []
                    self.sequence_map[seq_key].append(full_img_path)

        # Filtra sequências com apenas 1 imagem e cria lista de chaves
        self.sequence_map = {k: v for k, v in self.sequence_map.items() if len(v) > 1}
        self.valid_keys = list(self.sequence_map.keys())
        print(f"Dataset pronto: {len(self.valid_keys)} sequências válidas.")

    def __len__(self):
        return len(self.valid_keys)

    def __getitem__(self, idx):
        anchor_seq = self.valid_keys[idx]
        imgs = self.sequence_map[anchor_seq]
        
        # Estratégia simples de mineração
        anchor_path = random.choice(imgs)
        positive_path = random.choice(imgs)
        while positive_path == anchor_path and len(imgs) > 1:
            positive_path = random.choice(imgs)
            
        neg_seq = random.choice(self.valid_keys)
        while neg_seq == anchor_seq:
            neg_seq = random.choice(self.valid_keys)
        negative_path = random.choice(self.sequence_map[neg_seq])
        
        return self._load(anchor_path), self._load(positive_path), self._load(negative_path)

    def _load(self, path):
        return self.transform(Image.open(path).convert('RGB'))

# --- MODELO VPR ---
class VPRModel(nn.Module):
    def __init__(self):
        super(VPRModel, self).__init__()
        # Backbone: ResNet18 (leve para testar) ou ResNet50
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Removemos o FC original e AvgPool para acesso direto aos features
        # Mas para simplificar, vamos usar a saída do FC modificado
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]) # Remove fc layer
        self.fc = nn.Linear(512, 512) # Projeção
        
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc(x)
        # CRÍTICO: Normalização L2
        # Isso garante que todos os vetores vivam numa hiperesfera unitária
        return F.normalize(x, p=2, dim=1)

# --- EXECUÇÃO ---
if __name__ == '__main__':
    # 1. Configurações
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 2. Dataset e DataLoader
    # Ajuste o root_dir conforme sua estrutura real
    dataset = MSLSTripletDataset(root_dir='.', split='train_val', transform=transform)
    
    # Num_workers > 0 requer o bloco if __name__ == '__main__' no Windows
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    # 3. Inicialização
    model = VPRModel().to(device)
    criterion = nn.TripletMarginLoss(margin=0.6, p=2)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 4. Loop de Treino
    print("Iniciando treinamento...")
    model.train()

    epochs = 3
    for epoch in range(epochs):
        total_loss = 0
        for i, (anchor, positive, negative) in enumerate(dataloader):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            ea = model(anchor)
            ep = model(positive)
            en = model(negative)
            
            # Loss
            loss = criterion(ea, ep, en)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 10 == 0:
                print(f"Epoch {epoch} | Batch {i} | Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"===> Fim Epoch {epoch} | Média Loss: {avg_loss:.4f}")

    # 5. Salvar
    torch.save(model.state_dict(), "vpr_resnet18_l2.pth")
    print("Modelo salvo.")
