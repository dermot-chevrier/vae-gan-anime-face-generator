import torch
from torchvision.utils import save_image
from model import Generator
import os

# Parameter
LATENT_DIM = 100
NUM_IMAGES = 25  # Anzahl der zu generierenden Bilder
MODEL_PATH = "generator.pth"
OUTPUT_DIR = "generated_samples"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Aktives Gerät: {device}")

# Output-Ordner anlegen
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Generator laden
generator = Generator(latent_dim=LATENT_DIM).to(device)
generator.load_state_dict(torch.load(MODEL_PATH, map_location=device))
generator.eval()
print(f"✅ Generator-Modell aus '{MODEL_PATH}' geladen.")

# Noise generieren
noise = torch.randn(NUM_IMAGES, LATENT_DIM, 1, 1, device=device)

# Bilder generieren
with torch.no_grad():
    fake_images = generator(noise).detach().cpu()

# Bilder speichern
save_image(fake_images, os.path.join(OUTPUT_DIR, "generated_batch.png"), nrow=5, normalize=True)
print(f"{NUM_IMAGES} Bilder im Ordner '{OUTPUT_DIR}' gespeichert.")
