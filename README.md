# 🧙‍♂️ GeoWarlock: Geolocalização Multi-Modal via Imagens de Rua e OCR

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![Task](https://img.shields.io/badge/Task-Image_Classification-green)
![Status](https://img.shields.io/badge/Status-Em_Desenvolvimento-yellow)

> **GeoWarlock** é um modelo de Deep Learning multi-modal projetado para classificar a cidade de origem de imagens em nível de rua. O sistema combina análise visual (arquitetura, flora) com dados textuais (sinalização, idioma) para realizar geolocalização precisa, inspirado na mecânica do jogo *GeoGuessr*.

---

## 👥 Autores
* **Augusto Mattei Grohmann**
* **Angelo Fernandes Oliveira**

---

## 🎯 Objetivo
Desenvolver um sistema capaz de prever a cidade de origem de uma imagem (dentre 30 classes alvo) utilizando uma abordagem híbrida:
1.  **Visual:** Fine-tuning de um Vision Transformer (ViT) para capturar padrões visuais globais.
2.  **Textual:** Pipeline de OCR (Optical Character Recognition) para extrair textos de placas e fachadas, detectando o idioma predominante como uma feature auxiliar ("nudge").

O objetivo final é comparar a performance de um modelo puramente visual contra a abordagem multi-modal, demonstrando como o contexto linguístico auxilia na desambiguidade de locais visualmente similares.

---

## 🧠 Arquitetura do Modelo

O projeto é dividido em dois pipelines principais que convergem para a classificação final:

### 1. Pipeline Visual (Backbone)
* **Modelo:** Vision Transformer (ViT-Base/16) pré-treinado.
* **Processamento:** As imagens são redimensionadas e normalizadas.
* **Saída:** Vetor de features visuais (embeddings).

### 2. Pipeline Textual (Auxiliar)
* **OCR:** Utilização de biblioteca de reconhecimento óptico (ex: EasyOCR) para varrer o dataset.
* **Detecção de Idioma:** Classificação do texto extraído (ex: `langdetect`) para identificar a língua predominante.
* **Encoding:** O idioma é convertido em um vetor (One-Hot Encoding ou similar).

### 3. Fusão Multi-Modal
* Concatenação dos vetores Visual e Textual.
* Processamento por um perceptron multicamadas (MLP Head).
* Saída final: Softmax para 30 cidades.

---

## 📂 Dataset
Utilizamos o **Mapillary Street-level Sequences (MSLS) Dataset**.
* **Escopo:** Subconjunto de dados abrangendo **30 cidades** espalhadas por 6 continentes.
* **Volume:** Seleção curada para viabilizar o treinamento em tempo hábil (focado em experimentação acadêmica).

---

## 🚀 Como Executar (Em breve)

### Pré-requisitos
* Python 3.10+
* CUDA compatível com PyTorch (Recomendado GPU com 12GB+ VRAM para treino rápido)

### Instalação
```bash
# Clone o repositório
git clone [https://github.com/seu-usuario/geowarlock.git](https://github.com/seu-usuario/geowarlock.git)

# Instale as dependências
pip install -r requirements.txt