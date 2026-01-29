import torch
from torchvision.utils import save_image
from model import VAE
from datetime import datetime


# Settings
LATENT_DIM = 256
NUM_IMAGES = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_PATH = f"outputs/generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
CHECKPOINT = "checkpoints/vae_final.pth"

# 1. Load the model
vae = VAE(latent_dim=LATENT_DIM).to(DEVICE)
vae.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
vae.eval()

# 2. Sample random latent vectors
with torch.no_grad():
    z = torch.randn(NUM_IMAGES, LATENT_DIM).to(DEVICE)

    # 3. Decode vectors into images
    generated = vae.decoder(z)

    # 4. Un-normalize from [-1, 1] to [0, 1]
    #generated = (generated + 1) / 2          only to activeate when using MSE

    # 5. Save as image grid
    save_image(generated, OUTPUT_PATH, nrow=4)

print(f" Generated {NUM_IMAGES} new anime faces to {OUTPUT_PATH}")
