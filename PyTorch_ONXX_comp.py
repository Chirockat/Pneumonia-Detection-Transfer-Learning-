import time
import torch
import torchvision
import onnxruntime as ort
import numpy as np
import PIL.Image as Image
from torchvision import transforms

# ==========================================
# 1. TEST CONFIGURATION
# ==========================================
image_paths = [
    r"img\NORMAL-3983280-0001.jpeg",
    r"img\BACTERIA-840611-0001.jpeg",
    r"img\VIRUS-1540910-0001.jpeg"
]

# Transforms must be 100% identical to your val_transforms
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

classes = ['NORMAL', 'BACTERIA', 'VIRUS']

# ==========================================
# 2. LOADING ENGINES (PYTORCH & ONNX)
# ==========================================
print("Loading PyTorch model (CPU)...")
pytorch_model = torchvision.models.efficientnet_b0()
pytorch_model.classifier[1] = torch.nn.Linear(pytorch_model.classifier[1].in_features, 3)
pytorch_model.load_state_dict(torch.load('model_best.pth', map_location=torch.device('cpu'), weights_only=True))
pytorch_model.eval()

print("Loading ONNX model (CPU)...")
# Forcing CPUExecutionProvider to ensure a fair benchmark (both models running on CPU)
session = ort.InferenceSession("chest_xray_model.onnx", providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

# ==========================================
# 3. WARM-UP
# ==========================================
# Passing a dummy tensor to load backend libraries into RAM before measuring time
print("Warming up engines...")
dummy_input = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    pytorch_model(dummy_input)
session.run(None, {input_name: dummy_input.numpy()})


# Helper function to calculate Softmax probabilities
def get_probabilities(logits):
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / exp_logits.sum()


# ==========================================
# 4. MAIN BENCHMARK LOOP
# ==========================================
print("\n" + "=" * 50)
print("STARTING INFERENCE BENCHMARK")
print("=" * 50)

for path in image_paths:
    true_label = path.split(chr(92))[-1].split('-')[0]
    print(f"\nEvaluating Image: {true_label}")

    img = Image.open(path).convert('RGB')
    input_tensor = preprocess(img).unsqueeze(0)
    input_numpy = input_tensor.numpy()

    # --- PYTORCH INFERENCE ---
    t0 = time.perf_counter()
    with torch.no_grad():
        pt_outputs = pytorch_model(input_tensor)
    t1 = time.perf_counter()
    pt_time_ms = (t1 - t0) * 1000

    # --- ONNX INFERENCE ---
    t0 = time.perf_counter()
    onnx_outputs = session.run(None, {input_name: input_numpy})
    t1 = time.perf_counter()
    onnx_time_ms = (t1 - t0) * 1000

    # --- CALCULATIONS & ANALYSIS ---
    pt_logits = pt_outputs[0].numpy()
    onnx_logits = onnx_outputs[0][0]

    pt_probs = get_probabilities(pt_logits)
    onnx_probs = get_probabilities(onnx_logits)

    pt_pred_class = classes[np.argmax(pt_probs)]
    onnx_pred_class = classes[np.argmax(onnx_probs)]

    pt_conf = np.max(pt_probs) * 100
    onnx_conf = np.max(onnx_probs) * 100

    print(f"  PyTorch Result: {pt_pred_class:8} ({pt_conf:5.2f}%) | Time: {pt_time_ms:6.2f} ms")
    print(f"  ONNX Result:    {onnx_pred_class:8} ({onnx_conf:5.2f}%) | Time: {onnx_time_ms:6.2f} ms")