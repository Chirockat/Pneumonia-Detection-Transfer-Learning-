import torch_directml

from model_train import  ChestXrayDataset

import kagglehub
import torch
import torch.nn as nn
from torchvision import models, transforms
from tqdm import tqdm

dataset_path = kagglehub.dataset_download("tolgadincer/labeled-chest-xray-images")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

### Data (Only Val) ###
dataset = ChestXrayDataset(dataset_path, transform=transform)

torch.manual_seed(42)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

_, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True)

### Model ###
model = models.efficientnet_b0() # Nie mamy weights, co oznacza ze
# model jest wypelniony calkowicie losowymi liczbami

num_ftrs  = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 3)

model.load_state_dict(torch.load("model_wages.pth", weights_only=False))
print("Model weights loaded successfully.")

### Testing ###



correct = 0
total = 0

with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="Evaluating"):
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Validation Accuracy: {100 * correct / total:.2f}%")