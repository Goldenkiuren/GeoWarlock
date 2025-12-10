import os
import threading
import tkinter as tk
from tkinter import filedialog
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models
import customtkinter as ctk

MODEL_PATHS = {
    "ViT": "best_models/vit16_best.pth",
    "Dino Frozen": "best_models/frozen_dinov2_best.pth",
    "Dino Tuned": "best_models/dinov2_best.pth"
}

CLASS_NAMES = [
    'amsterdam', 'athens', 'austin', 'bangkok', 'bengaluru', 'berlin', 'boston', 
    'budapest', 'buenosaires', 'cph', 'goa', 'helsinki', 'kampala', 'london', 
    'manila', 'melbourne', 'miami', 'moscow', 'ottawa', 'paris', 'phoenix', 
    'saopaulo', 'sf', 'stockholm', 'tokyo', 'toronto', 'trondheim', 'zurich'
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DinoV2FineTuning(nn.Module):
    def __init__(self, num_classes):
        super(DinoV2FineTuning, self).__init__()
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

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

class GeoWarlockApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("GeoWarlock")
        self.geometry("950x700")
        
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.current_image_path = None
        self.loaded_models = {}
        self.num_classes = len(CLASS_NAMES)

        self.sidebar_frame = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="GeoWarlock", font=ctk.CTkFont(size=22, weight="bold"))
        self.logo_label.pack(padx=20, pady=(30, 20))

        self.btn_load = ctk.CTkButton(self.sidebar_frame, text="Load Image", command=self.select_image)
        self.btn_load.pack(padx=20, pady=10)

        self.divider = ctk.CTkFrame(self.sidebar_frame, height=2, fg_color="gray30")
        self.divider.pack(padx=20, pady=20, fill="x")

        self.lbl_model = ctk.CTkLabel(self.sidebar_frame, text="Select Arcana (Model):", anchor="w")
        self.lbl_model.pack(padx=20, pady=(0, 10), anchor="w")

        self.model_var = ctk.StringVar(value="ViT")
        models_list = ["ViT", "Dino Frozen", "Dino Tuned", "All 3"]
        
        for m in models_list:
            rb = ctk.CTkRadioButton(self.sidebar_frame, text=m, variable=self.model_var, value=m)
            rb.pack(padx=20, pady=8, anchor="w")

        ctk.CTkLabel(self.sidebar_frame, text="").pack(pady=10)

        self.btn_run = ctk.CTkButton(
            self.sidebar_frame, 
            text="Guess City", 
            fg_color="#7B2CBF", hover_color="#5A189A", 
            command=self.start_inference
        )
        self.btn_run.pack(padx=20, pady=20)
        
        self.status_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.status_frame.pack(side="bottom", fill="x", pady=20)
        self.status_label = ctk.CTkLabel(self.status_frame, text="Ready", text_color="gray", font=("Arial", 12))
        self.status_label.pack()

        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        self.main_frame.grid_rowconfigure(0, weight=4) 
        self.main_frame.grid_rowconfigure(1, weight=3)
        self.main_frame.grid_columnconfigure(0, weight=1)

        self.image_label = ctk.CTkLabel(
            self.main_frame, text="No Image Selected\n\nClick 'Load Image' to start", 
            width=400, height=350, corner_radius=10, fg_color="#2B2B2B"
        )
        self.image_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.results_bg = ctk.CTkFrame(self.main_frame, corner_radius=10, fg_color="#2B2B2B")
        self.results_bg.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        
        self.results_bg.grid_rowconfigure(0, weight=1)
        self.results_bg.grid_columnconfigure(0, weight=1)

        self.results_label = ctk.CTkLabel(
            self.results_bg, 
            text="Results will appear here...", 
            font=("Consolas", 14),
            justify="left",
            anchor="center"
        )
        self.results_label.grid(row=0, column=0, padx=20, pady=20)

    def select_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png"), ("All Files", "*.*")]
        )
        if file_path:
            self.current_image_path = file_path
            pil_img = Image.open(file_path)
            w, h = pil_img.size
            aspect = w / h
            target_h = 350
            target_w = int(target_h * aspect)
            
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(target_w, target_h))
            self.image_label.configure(text="", image=ctk_img)
            self.status_label.configure(text="Image Loaded")

    def get_model_instance(self, model_name):
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        path = MODEL_PATHS.get(model_name)
        if not path or not os.path.exists(path):
            return None

        self.status_label.configure(text=f"Loading {model_name}...")
        self.update()

        try:
            if model_name == "Dino Tuned":
                model = DinoV2FineTuning(self.num_classes)
            elif model_name == "ViT":
                model = ViTForCityClassification(self.num_classes)
            elif model_name == "Dino Frozen":
                model = DinoV2ForCityClassification(self.num_classes)
            
            model.to(DEVICE)
            checkpoint = torch.load(path, map_location=DEVICE)
            new_state_dict = {}
            for k, v in checkpoint.items():
                new_key = k.replace("_orig_mod.", "")
                new_state_dict[new_key] = v
            
            model.load_state_dict(new_state_dict)
            model.eval()
            self.loaded_models[model_name] = model
            return model
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            return None

    def start_inference(self):
        if not self.current_image_path:
            self.status_label.configure(text="Please load an image first!", text_color="#FF5555")
            return
        threading.Thread(target=self.run_inference, daemon=True).start()

    def run_inference(self):
        self.btn_run.configure(state="disabled", text="Casting...")
        self.status_label.configure(text="Processing...", text_color="white")
        
        selection = self.model_var.get()

        if selection == "All 3":
            models_to_run = ["ViT", "Dino Frozen", "Dino Tuned"]
        else:
            models_to_run = [selection]
            
        results_text = ""
        
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        try:
            img = Image.open(self.current_image_path).convert("RGB")
            input_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
            
            for m_name in models_to_run:
                model = self.get_model_instance(m_name)
                if model is None:
                    results_text += f"[{m_name}] - Model not found\n"
                    continue

                with torch.no_grad():
                    logits = model(input_tensor)
                    probs = torch.nn.functional.softmax(logits, dim=1)
                    top3_prob, top3_idx = torch.topk(probs, 3)
                    
                    results_text += f"=== {m_name} ===\n"
                    for i in range(3):
                        city = CLASS_NAMES[top3_idx[0][i].item()]
                        prob = top3_prob[0][i].item() * 100
                        results_text += f"{i+1}. {city.upper()}: {prob:.1f}%\n"
                    
                    if m_name != models_to_run[-1]:
                        results_text += "\n"

            self.results_label.configure(text=results_text)
            self.status_label.configure(text="Complete", text_color="#2CC985")

        except Exception as e:
            self.status_label.configure(text=f"Error: {str(e)}", text_color="#FF5555")
            print(e)
        finally:
            self.btn_run.configure(state="normal", text="Guess City")

if __name__ == "__main__":
    app = GeoWarlockApp()
    app.mainloop()