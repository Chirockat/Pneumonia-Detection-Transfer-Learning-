import os
import sys
import tkinter as tk
from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import Image, ImageTk
import io
import numpy as np
import onnxruntime as ort


def resource_path(relative_path):
    """
    Get absolute path to resource, works for dev and for PyInstaller.
    PyInstaller creates a temp folder and stores path in _MEIPASS.
    """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def preprocess_image(img_path):
    """NumPy recreation of PyTorch transforms to avoid heavy PyTorch dependency."""
    img = Image.open(img_path).convert('RGB')
    img_resized = img.resize((224, 224), resample=Image.Resampling.BILINEAR)

    img_np = np.array(img_resized).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_np = (img_np - mean) / std

    img_np = np.transpose(img_np, (2, 0, 1))
    img_np = np.expand_dims(img_np, axis=0)

    return img_np, img


def get_probabilities(logits):
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / exp_logits.sum()


class ChestXrayStandaloneApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chest X-Ray Pneumonia Detector (Standalone)")
        self.root.geometry("600x550")

        self.bg_color = "#f5f6fa"
        self.drop_color = "#dcdde1"
        self.highlight_color = "#dff9fb"
        self.text_color = "#2f3640"
        self.root.configure(bg=self.bg_color)

        self.photo = None
        self.classes = ['NORMAL', 'BACTERIA', 'VIRUS']

        # --- LOAD ONNX MODEL ---
        try:
            model_path = resource_path("chest_xray_model.onnx")
            self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            self.input_name = self.session.get_inputs()[0].name
            self.model_loaded = True
        except Exception as e:
            self.model_loaded = False
            print(f"Failed to load model: {e}")

        self.setup_ui()

    def setup_ui(self):
        self.header = tk.Label(self.root, text="Pneumonia Detection System", font=("Arial", 16, "bold"),
                               bg=self.bg_color, fg=self.text_color)
        self.header.pack(pady=10)

        self.drop_frame = tk.Frame(self.root, bg=self.drop_color, bd=2, relief="groove", cursor="hand2")
        self.drop_frame.pack(fill="both", expand=True, padx=40, pady=10)

        self.drop_label = tk.Label(self.drop_frame,
                                   text="[ Drag & Drop X-Ray Image Here ]\n(Supports: JPEG, PNG, BMP, TIFF)",
                                   font=("Arial", 11), bg=self.drop_color, fg="#718093", justify="center")
        self.drop_label.place(relx=0.5, rely=0.5, anchor="center")

        self.drop_frame.drop_target_register(DND_FILES)
        self.drop_frame.dnd_bind('<<Drop>>', self.handle_drop)
        self.drop_frame.dnd_bind('<<DragEnter>>', lambda e: self.drop_frame.config(bg=self.highlight_color))
        self.drop_frame.dnd_bind('<<DragLeave>>', lambda e: self.drop_frame.config(bg=self.drop_color))

        self.result_frame = tk.Frame(self.root, bg=self.bg_color)
        self.result_frame.pack(fill="x", padx=40, pady=10)

        self.lbl_filename = tk.Label(self.result_frame, text="File: -", font=("Arial", 10), bg=self.bg_color,
                                     anchor="w", fg=self.text_color)
        self.lbl_filename.pack(fill="x")

        self.lbl_details = tk.Label(self.result_frame, text="NORMAL: - | BACTERIA: - | VIRUS: -", font=("Arial", 10),
                                    bg=self.bg_color, anchor="w", fg="#718093")
        self.lbl_details.pack(fill="x", pady=5)

        self.lbl_score = tk.Label(self.result_frame, text="READY", font=("Arial", 20, "bold"), bg=self.bg_color,
                                  fg="#718093")
        self.lbl_score.pack(pady=5)

        if not self.model_loaded:
            self.lbl_score.config(text="ERROR: MODEL NOT FOUND", fg="red")

    def handle_drop(self, event):
        if not self.model_loaded:
            return

        self.drop_frame.config(bg=self.drop_color)
        filepath = event.data

        if filepath.startswith('{') and filepath.endswith('}'):
            filepath = filepath[1:-1]

        filename = os.path.basename(filepath)
        self.lbl_filename.config(text=f"File: {filename}")
        self.lbl_score.config(text="Analyzing...", fg="#718093")
        self.root.update_idletasks()

        try:
            # Preprocess
            input_tensor, original_img = preprocess_image(filepath)

            # GUI Image Preview
            preview_img = original_img.copy()
            preview_img.thumbnail((350, 250))
            self.photo = ImageTk.PhotoImage(preview_img)
            self.drop_label.config(image=self.photo, text="")

            # ONNX Inference
            outputs = self.session.run(None, {self.input_name: input_tensor})
            logits = outputs[0][0]

            probs = get_probabilities(logits)
            predicted_idx = np.argmax(probs)

            diagnosis = self.classes[predicted_idx]
            confidence = probs[predicted_idx] * 100

            # Update GUI
            self.lbl_details.config(
                text=f"NORMAL: {probs[0] * 100:.1f}% | BACTERIA: {probs[1] * 100:.1f}% | VIRUS: {probs[2] * 100:.1f}%")
            self.lbl_score.config(text=f"{diagnosis} ({confidence:.1f}%)")

            if diagnosis == "NORMAL":
                self.lbl_score.config(fg="#27ae60")
            elif diagnosis == "VIRUS":
                self.lbl_score.config(fg="#e67e22")
            else:
                self.lbl_score.config(fg="#c0392b")

        except Exception as e:
            self.lbl_score.config(text=f"Error processing image", fg="red")
            print(e)


if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = ChestXrayStandaloneApp(root)
    root.mainloop()