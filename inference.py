import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# --- CONFIGURATION ---
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ALPHABETICAL list of your 30 cities
CLASSES = [
    'amsterdam', 'athens', 'austin', 'bangkok', 'bengaluru', 
    'berlin', 'boston', 'budapest', 'buenosaires', 'cph', 'goa', 
    'helsinki', 'kampala', 'london', 'manila', 'melbourne', 'miami', 
    'moscow', 'ottawa', 'paris', 'phoenix', 'saopaulo', 
    'sf', 'stockholm', 'tokyo', 'toronto', 'trondheim', 'zurich'
]

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

def predict_image(image_path, model):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ])
    
    try:
        image = Image.open(image_path).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            top_probs, top_idxs = torch.topk(probs, 3)
            
        print(f"\n--- Result for {os.path.basename(image_path)} ---")
        for i in range(3):
            city = CLASSES[top_idxs[0][i].item()]
            confidence = top_probs[0][i].item() * 100
            print(f"  {i+1}. {city.upper()}: {confidence:.1f}%")
            
    except Exception as e:
        print(f"Error: {e}")

def main():
    print(f"Loading model on {DEVICE}...")
    model = ViTForCityClassification(num_classes=len(CLASSES)).to(DEVICE)
    
    # Load your trained weights
    if not os.path.exists("best_city_model.pth"):
        print("Error: best_city_model.pth not found!")
        return

    state_dict = torch.load("best_city_model.pth", map_location=DEVICE, weights_only=True)
    
    # Remove 'torch.compile' prefix if present (common issue)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("_orig_mod.", "")
        new_state_dict[new_key] = v
        
    model.load_state_dict(new_state_dict)
    model.eval()
    print("Model loaded! Ready to test.")

    print("\nPaste path to an image (or type 'q' to quit)")
    while True:
        path = input("Image Path: ").strip().strip('"').strip("'")
        if path.lower() == 'q': break
        if os.path.exists(path): predict_image(path, model)
        else: print("File not found.")

if __name__ == "__main__":
    main()