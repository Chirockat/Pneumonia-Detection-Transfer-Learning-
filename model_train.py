import kagglehub
import glob
import os
import PIL.Image as Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch_directml
import torchvision


from torchvision import transforms

dataset_path = kagglehub.dataset_download("tolgadincer/labeled-chest-xray-images")
print(dataset_path)

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ChestXrayDataset(torch.utils.data.Dataset):
    def __init__ (self, root_dir, transform=None):
        self.image_paths = glob.glob(os.path.join(root_dir, '**/*.jpeg'), recursive=True)
        self.class_to_idx = {'NORMAL': 0, 'BACTERIA': 1, 'VIRUS': 2}

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        filename = os.path.basename(image_path)
        label_str = filename.split('-')[0]
        label = self.class_to_idx[label_str]

        if self.transform:
            image = self.transform(image)

        return image, label



# Przykladowe wypisanie
"""
dataset = ChestXrayDataset(dataset_path)

image, label = dataset[6]
plt.imshow(image)
plt.title(f"Etykieta klasy: {label}")
plt.axis('off')
plt.show()
"""

# Ciekawostka — po tej operacji juz nie zadziala
# wyswietlenie obrazu, bo z 224, 224, 3
# zmienilo sie na 3, 224, 224

if __name__ == '__main__':
    ### Data Split and Data Loader ###
    dataset = ChestXrayDataset(dataset_path, transform=data_transforms)

    torch.manual_seed(42)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    ### Model Effiecent-Net-B0, Fine-Tuning, Training 2 last Blocks ###

    # To jeszcze do zmiany weights
    model = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_ftrs, 3)

    # Jak wygladaja nasze warstwy:
    """
    print("\n--- Struktura Głównego Ciała Modelu (Features) ---")
    for i, block in enumerate(model.features):
        # Pobieramy pierwszy parametr z bloku, żeby sprawdzić jego status
        is_trainable = next(block.parameters()).requires_grad
        # block.__class__.__name__ poda nam techniczną nazwę warstwy
        print(f"Blok {i}: Gotowy do nauki = {is_trainable} | Typ: {block.__class__.__name__}")

    print("\n--- Głowa Decyzyjna (Classifier) ---")
    for i, layer in enumerate(model.classifier):
        # Dropout nie ma wag, więc musimy to bezpiecznie ominąć
        params = list(layer.parameters())
        if params:
            is_trainable = params[0].requires_grad
        else:
            is_trainable = "Brak wag (np. Dropout)"

        print(f"Warstwa {i}: Gotowa do nauki = {is_trainable} | Typ: {layer.__class__.__name__}")
    """

    # Funkcja straty
    criterion = torch.nn.CrossEntropyLoss()



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch_directml.is_available():
        device = torch_directml.device()
        print(f"Sukces! Znaleziono kartę: {torch_directml.device_name(device.index)}")

    model.to(device)
    print(f"Training on: {device}")


    optimizer = torch.optim.Adam(model.classifier[1].parameters(), lr=0.001)

    num_epochs = 5

    for epoch in range(num_epochs):

        # ================================
        # 1. FAZA TRENINGU (Uczenie się)
        # ================================

        if epoch == 2:
            print("\n[INFO] Epoka 3: Odmrażam głębokie warstwy modelu (Fine-Tuning)!")

            # Odmrażamy dwa ostatnie grube bloki
            for param in model.features[-2:].parameters():
                param.requires_grad = True

            trainable_params = filter(lambda p: p.requires_grad, model.parameters())

            optimizer = torch.optim.Adam(trainable_params, lr=0.0001)

        model.train()
        running_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        # ================================
        # 2. FAZA WALIDACJI (Testowanie)
        # ================================
        model.eval()
        val_correct = 0
        val_total = 0

        val_progress = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]")

        with torch.no_grad():
            for val_images, val_labels in val_progress:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                outputs = model(val_images)

                _, predicted = torch.max(outputs, 1)
                val_total += val_labels.size(0)
                val_correct += (predicted == val_labels).sum().item()

        # ================================
        # 3. PODSUMOWANIE EPOKI
        # ================================
        train_loss = running_loss / len(train_loader)
        val_acc = 100 * val_correct / val_total

        print(f"--> Summary | Train Loss: {train_loss:.4f} | Val Accuracy: {val_acc:.2f}%\n")

        # Zapisujemy same wagi do pliku z rozszerzeniem .pth (standard PyTorcha)
        torch.save(model.state_dict(), 'model_wages_v2.pth')
        print("Saved as: model_wages.pth")

