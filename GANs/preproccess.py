from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

IMAGE_SIZE = 64
BATCH_SIZE = 128
DATA_DIR = './data/anime_faces'

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # → [-1, 1]
])

full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)

train_size = int(0.9 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
