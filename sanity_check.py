import onnxruntime as ort
import numpy as np
import PIL.Image as Image
from torchvision import transforms

# ==========================================
# 1. KONFIGURACJA TESTU
# ==========================================
# Wklej tutaj ścieżkę do dowolnego zdjęcia z folderu Kaggle
#image_path = r"C:\Users\zeitu\.cache\kagglehub\datasets\tolgadincer\labeled-chest-xray-images\versions\1\chest_xray\test\NORMAL\NORMAL-3983280-0001.jpeg"
#image_path = r"C:\Users\zeitu\.cache\kagglehub\datasets\tolgadincer\labeled-chest-xray-images\versions\1\chest_xray\test\PNEUMONIA\BACTERIA-840611-0001.jpeg"
image_path = r"C:\Users\zeitu\.cache\kagglehub\datasets\tolgadincer\labeled-chest-xray-images\versions\1\chest_xray\test\PNEUMONIA\VIRUS-1540910-0001.jpeg"

# Transformacje muszą być w 100% identyczne jak w Twoim val_transforms!
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 2. URUCHOMIENIE SILNIKA ONNX
# ==========================================
print("Loading ONNX model...")
session = ort.InferenceSession("chest_xray_model.onnx")

# Przygotowanie zdjęcia
img = Image.open(image_path).convert('RGB')
input_tensor = preprocess(img)

# ONNX Runtime nie rozumie Tensorów PyTorcha, musimy zamienić to na czystą matematykę (Numpy)
# Dodajemy też jeden wymiar na początku (tzw. batch_size = 1), bo model uczył się na paczkach
input_batch = input_tensor.unsqueeze(0).numpy()

# ==========================================
# 3. PREDYKCJA
# ==========================================
input_name = session.get_inputs()[0].name
# Wysyłamy zdjęcie do czarnej skrzynki
outputs = session.run(None, {input_name: input_batch})

# Wyciągamy surowe wyniki (tzw. logits)
logits = outputs[0][0]

# ==========================================
# 4. ANALIZA WYNIKÓW
# ==========================================
classes = ['NORMAL', 'BACTERIA', 'VIRUS']

# Matematyczna funkcja Softmax - zamienia dziwne liczby modelu na ładne wartości procentowe
exp_logits = np.exp(logits - np.max(logits))
probabilities = exp_logits / exp_logits.sum()

predicted_idx = np.argmax(probabilities)
predicted_class = classes[predicted_idx]
confidence = probabilities[predicted_idx] * 100

print(f"\n--- DIAGNOSIS RESULT ---")
print(f"File name indicates: {image_path.split(chr(92))[-1].split('-')[0]}") # Wyciąga info z nazwy pliku
print(f"Model prediction:    {predicted_class}")
print(f"Confidence:          {confidence:.2f}%")

print("\nDetailed probabilities:")
for i, cls in enumerate(classes):
    print(f"{cls}: {probabilities[i]*100:.2f}%")