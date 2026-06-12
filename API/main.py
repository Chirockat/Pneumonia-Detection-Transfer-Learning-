from fastapi import FastAPI, UploadFile, File, HTTPException
import onnxruntime as ort
import numpy as np
from PIL import Image
import io

app = FastAPI(title="Chest X-Ray Pneumonia API")

# Load the ONNX model into memory when the server starts
print("Loading ONNX model...")
try:
    session = ort.InferenceSession("chest_xray_model.onnx", providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

CLASSES = ['NORMAL', 'BACTERIA', 'VIRUS']


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Pure NumPy recreation of PyTorch transforms (Resize, ToTensor, Normalize).
    This saves ~2GB of Docker image space by eliminating the PyTorch dependency.
    """
    # Load and convert to RGB
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    # 1. Resize (224, 224)
    img = img.resize((224, 224))

    # 2. ToTensor (convert to float and scale 0-1)
    img_np = np.array(img).astype(np.float32) / 255.0

    # 3. Normalize
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_np = (img_np - mean) / std

    # 4. Change dimension order from HWC to CHW (PyTorch standard)
    img_np = np.transpose(img_np, (2, 0, 1))

    # 5. Add batch dimension -> shape becomes (1, 3, 224, 224)
    img_np = np.expand_dims(img_np, axis=0)

    return img_np


def get_probabilities(logits: np.ndarray) -> np.ndarray:
    """Mathematical Softmax function."""
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / exp_logits.sum()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        # Read file bytes
        image_bytes = await file.read()

        # Preprocess
        input_tensor = preprocess_image(image_bytes)

        # Run ONNX inference
        outputs = session.run(None, {input_name: input_tensor})
        logits = outputs[0][0]

        # Calculate probabilities
        probs = get_probabilities(logits)
        predicted_idx = np.argmax(probs)

        # Prepare response
        return {
            "filename": file.filename,
            "diagnosis": CLASSES[predicted_idx],
            "confidence": float(np.max(probs) * 100),
            "details": {
                "NORMAL": float(probs[0] * 100),
                "BACTERIA": float(probs[1] * 100),
                "VIRUS": float(probs[2] * 100)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))