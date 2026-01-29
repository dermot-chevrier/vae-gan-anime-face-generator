import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import Generator, Discriminator
import os
from torchvision.utils import save_image

# Device-Check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n✅ Aktives Gerät: {device}")
if device.type == 'cpu':
    print("⚠️  Achtung: CUDA nicht verfügbar.\n")

# Datenpfad
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'anime_faces')

# Parameter
IMAGE_SIZE = 64
BATCH_SIZE = 128
LATENT_DIM = 100
EPOCHS = 400  # Erhöht
LEARNING_RATE = 0.0002
BETA1 = 0.5
LABEL_SMOOTH_REAL = 0.9
NOISE_STD = 0.05  # Reduziert

# Transform
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Dataset & Loader
dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Modelle
generator = Generator(latent_dim=LATENT_DIM).to(device)
discriminator = Discriminator().to(device)

# Loss & Optimizer
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))

# Ordner für Ergebnisse
os.makedirs("generated_images", exist_ok=True)

# Fester Noise-Vektor für Visualisierung
fixed_noise = torch.randn(64, LATENT_DIM, 1, 1, device=device)

# Training
for epoch in range(EPOCHS):
    for i, (imgs, _) in enumerate(dataloader):

        real_imgs = imgs.to(device)

        noise = torch.randn_like(real_imgs) * NOISE_STD   # noise hinzu
        noisy_real_imgs = torch.clamp(real_imgs + noise, -1.0, 1.0)

        real_labels = torch.full((imgs.size(0), 1), LABEL_SMOOTH_REAL, device=device) # labels
        fake_labels = torch.zeros(imgs.size(0), 1, device=device)

        optimizer_D.zero_grad() # D |
                                #   V
        outputs_real = discriminator(noisy_real_imgs) #real
        d_loss_real = criterion(outputs_real, real_labels)

        z = torch.randn(imgs.size(0), LATENT_DIM, 1, 1, device=device) #fake
        fake_imgs = generator(z)

        outputs_fake = discriminator(fake_imgs.detach())
        d_loss_fake = criterion(outputs_fake, fake_labels)

        d_loss = d_loss_real + d_loss_fake #loss
        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad() # G |
                                #   V
        outputs = discriminator(fake_imgs)
        g_loss = criterion(outputs, real_labels)

        g_loss.backward()
        optimizer_G.step()

        if i % 100 == 0:  # console output
            print(f"Epoch [{epoch+1}/{EPOCHS}] Batch {i}/{len(dataloader)} "
                  f"Loss D: {d_loss.item():.4f}, Loss G: {g_loss.item():.4f}")

    with torch.no_grad(): #Bilder speichern
        sample_imgs = generator(fixed_noise).detach().cpu()
        save_image(sample_imgs, f"generated_images/epoch_{epoch+1}.png", nrow=8, normalize=True)

    print(f"Beispielbilder für Epoche {epoch+1} gespeichert.")

torch.save(generator.state_dict(), "generator.pth") # Model speichern
print("Generator-Modell gespeichert.")
