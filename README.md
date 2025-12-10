# 🧙‍♂️ GeoWarlock: Geolocalização Visual com Vision Transformers e DinoV2

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![DinoV2](https://img.shields.io/badge/Meta-DinoV2-green)
![Status](https://img.shields.io/badge/Status-Concluído-success)

> **GeoWarlock** é um sistema de Deep Learning desenvolvido para tarefas de geolocalização baseada em imagens (Image-to-GPS/City). O projeto realiza uma análise comparativa entre arquiteturas Vision Transformer clássicas (ViT-16) e modelos auto-supervisionados modernos (DinoV2) utilizando o dataset Mapillary Street-Level Sequences.

---

## 👥 Autores
* **Augusto Mattei Grohmann**
* **Angelo Fernandes Oliveira**

---

## 🎯 Objetivo e Metodologia
O objetivo deste projeto é classificar a cidade de origem de uma imagem de rua, mitigando problemas clássicos de *data leakage* em séries temporais de imagens.

### 🧠 Arquiteturas Comparadas
O projeto implementa e compara três abordagens distintas:
1.  **ViT-B/16 (Supervisionado):** Modelo pré-treinado na ImageNet-1k, servindo como baseline.
2.  **DinoV2 Frozen (Self-Supervised):** Utiliza o backbone do DinoV2 (ViT-B/14) com pesos congelados como extrator de características, treinando apenas o classificador (Head).
3.  **DinoV2 Fine-Tuned:** Ajuste fino completo do backbone DinoV2 e do classificador, com taxas de aprendizado diferenciais para preservar o conhecimento prévio.

### 🛡️ Estratégia de Treinamento
* **Split Espacial (K-Means):** Para evitar vazamento de dados (onde imagens da mesma rua aparecem no treino e validação), implementou-se uma divisão baseada em clusters de coordenadas GPS usando K-Means.
* **Balanceamento de Classes:** Utilização de `WeightedRandomSampler` para lidar com a disparidade no número de imagens entre cidades.
* **Data Augmentation:** Aplicação de `RandomResizedCrop`, `ColorJitter`, rotações e flips para forçar o modelo a aprender características estruturais e não apenas memorizar pixels.

---

## 📊 Resultados
O estudo, realizado com 28 classes (cidades), demonstrou a superioridade das abordagens auto-supervisionadas:

| Modelo | Acurácia Validação | Observações |
| :--- | :--- | :--- |
| **ViT-16** | ~83.5% | Apresentou sinais de overfitting após a 4ª época. |
| **DinoV2 Frozen** | ~92.6% | Melhor generalização em testes externos (Street View), com 100% de acerto na Europa. |
| **DinoV2 Tuned** | **>94%** | Maior acurácia bruta, porém com maior custo computacional. |

---

## 📂 Estrutura do Projeto

### Scripts de Treinamento
* `main_vit.py`: Pipeline de treinamento para o Vision Transformer clássico.
* `main_frozen_dino.py`: Treinamento do classificador linear sobre o backbone congelado do DinoV2.
* `main_dino.py`: Fine-tuning completo do DinoV2 com LR diferencial (Backbone: 5e-6, Head: 1e-4).

### Scripts de Inferência e Teste
* `geowarlock_city_guesser.py`: Aplicação GUI (Interface Gráfica) desenvolvida em CustomTkinter. Permite carregar uma imagem e obter predições em tempo real usando qualquer um dos três modelos[cite: 147, 148].
* `batch_test_*.py`: Scripts para avaliação em lote em pastas de teste, utilizando **Test-Time Augmentation (FiveCrop)** para aumentar a robustez da predição[cite: 108, 110].

---

## 🚀 Instalação e Uso

### Pré-requisitos

* Python 3.10+
* GPU com suporte a CUDA (recomendado para treino; CPU funciona para inferência mas é muito lento)
* Git (opcional, para clonar o repositório)

> **Recomendação:** crie um ambiente virtual antes de instalar dependências:

```bash
# Unix / macOS
python -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Instalação das dependências

```bash
# Instalar requisitos
pip install -r requirements.txt

# Ou instalar o pacote local (opcional)
pip install -e .
```

Se quiser reproduzir exatamente o ambiente, inclua um `requirements-lock.txt` ou um `environment.yml` (conda).

---

## 🔧 Executando a Interface Gráfica (GeoWarlock)

A GUI permite carregar uma imagem e obter a predição de cidade em tempo real.

```bash
# Exemplo: abrir a GUI usando um modelo específico
python geowarlock_city_guesser.py --model best_models/dino_finetuned.pth
```

**Observações:**
* Coloque os checkpoints `.pth` na pasta `best_models/` ou aponte com `--model` para o caminho correto.
* A GUI usa CustomTkinter; se houver problemas com o back-end, verifique a versão do `tkinter` e a compatibilidade da sua plataforma.

---

## 🏋️ Executando o Treinamento

Exemplo de comando para o fine-tuning do DinoV2 com taxas de aprendizado diferenciadas:

```bash
python main_dino.py \
  --data-dir data/ \
  --epochs 30 \
  --batch-size 64 \
  --lr-backbone 5e-6 \
  --lr-head 1e-4 \
  --output-dir runs/dino_finetuned \
  --seed 42
```

Para treinar apenas o classificador sobre o backbone congelado:

```bash
python main_frozen_dino.py --data-dir data/ --epochs 20 --batch-size 128 --output-dir runs/dino_frozen
```

E para o ViT baseline:

```bash
python main_vit.py --data-dir data/ --epochs 30 --batch-size 64 --output-dir runs/vit_baseline
```

> **Dica:** inclua `--resume` ou `--checkpoint` nos scripts para facilitar retomar treinamentos interrompidos.

---

## 🧪 Teste em Lote e Test-Time Augmentation

Exemplo de uso do script de avaliação em lote com TTA (FiveCrop):

```bash
python batch_test.py --model best_models/dino_frozen.pth --input-dir test_images/ --tta fivecrop --output results/batch_results.csv
```

---

## 📁 Estrutura esperada do dataset

O repositório assume a seguinte organização mínima do dataset (`data/`):

```
data/
├─ Amsterdam/
│  ├─ img_000001.jpg
│  └─ img_000002.jpg
├─ BuenosAires/
│  └─ ...
└─ SãoPaulo/
   └─ ...
```

* Cada pasta representa uma classe (cidade).
* No processamento que utilizamos, filtramos cidades com menos de 200 imagens e usamos splitting espacial (K-Means sobre coordenadas) para evitar data leakage entre treino/val.
* Se houver metadados (CSV com `filename,lat,lon,sequence_id`), descreva o formato esperado e onde colocá-lo (ex.: `data/metadata.csv`).

---

### 🌍 Dataset

O projeto utiliza uma versão curada do Mapillary Street-Level Sequences (MSLS). As classes incluem cidades como: Amsterdam, Buenos Aires, Tokyo, São Paulo, Paris, entre outras (totalizando 28 cidades após filtragem de classes com menos de 200 imagens).