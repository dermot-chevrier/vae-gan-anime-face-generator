import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
# If wanna change from MSE to BCE changge 1 thing in preproccess.py and 2 in model.py and 1 in train_vae.py

# Image size and batch size
IMAGE_SIZE = 64
BATCH_SIZE = 128
DATA_DIR = './data/anime_faces'

# Transform pipeline
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),  # Converts to [0, 1]
    #transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # -> [-1, 1] only activate this line when using MSE                            MSE/BCE
])

# Load dataset
full_dataset = datasets.ImageFolder(
    root=DATA_DIR,
    transform=transform
)

# If images are not in class subfolders, fix that later ----> if needed

# Train/test split
train_size = int(0.9 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# Create loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Loaded {len(train_dataset)} training and {len(test_dataset)} test images.")
