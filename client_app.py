import tkinter as tk
from tkinterdnd2 import DND_FILES, TkinterDnD
import os
import requests
from PIL import Image, ImageTk
import io


class ChestXrayClientApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chest X-Ray Pneumonia Detector")
        self.root.geometry("600x550")

        self.api_url = "http://127.0.0.1:8000/predict"

        self.bg_color = "#f5f6fa"
        self.drop_color = "#dcdde1"
        self.highlight_color = "#dff9fb"
        self.text_color = "#2f3640"
        self.root.configure(bg=self.bg_color)

        # Image reference to prevent garbage collection
        self.photo = None

        self.header = tk.Label(
            root,
            text="Pneumonia Detection System",
            font=("Arial", 16, "bold"),
            bg=self.bg_color,
            fg=self.text_color
        )
        self.header.pack(pady=10)

        self.drop_frame = tk.Frame(root, bg=self.drop_color, bd=2, relief="groove", cursor="hand2")
        self.drop_frame.pack(fill="both", expand=True, padx=40, pady=10)

        self.drop_label = tk.Label(
            self.drop_frame,
            text="[ Drag & Drop X-Ray Image Here ]\n(Supports: JPEG, PNG, BMP, TIFF)",
            font=("Arial", 11),
            bg=self.drop_color,
            fg="#718093",
            justify="center"
        )
        self.drop_label.place(relx=0.5, rely=0.5, anchor="center")

        self.drop_frame.drop_target_register(DND_FILES)
        self.drop_frame.dnd_bind('<<Drop>>', self.handle_drop)
        self.drop_frame.dnd_bind('<<DragEnter>>', lambda e: self.drop_frame.config(bg=self.highlight_color))
        self.drop_frame.dnd_bind('<<DragLeave>>', lambda e: self.drop_frame.config(bg=self.drop_color))

        self.result_frame = tk.Frame(root, bg=self.bg_color)
        self.result_frame.pack(fill="x", padx=40, pady=10)

        self.lbl_filename = tk.Label(self.result_frame, text="File: -", font=("Arial", 10), bg=self.bg_color,
                                     anchor="w", fg=self.text_color)
        self.lbl_filename.pack(fill="x")

        self.lbl_details = tk.Label(self.result_frame, text="NORMAL: - | BACTERIA: - | VIRUS: -", font=("Arial", 10),
                                    bg=self.bg_color, anchor="w", fg="#718093")
        self.lbl_details.pack(fill="x", pady=5)

        self.lbl_score = tk.Label(self.result_frame, text="DIAGNOSIS: -", font=("Arial", 20, "bold"), bg=self.bg_color,
                                  fg="#718093")
        self.lbl_score.pack(pady=5)

    def handle_drop(self, event):
        self.drop_frame.config(bg=self.drop_color)
        filepath = event.data

        if filepath.startswith('{') and filepath.endswith('}'):
            filepath = filepath[1:-1]

        filename = os.path.basename(filepath)
        self.lbl_filename.config(text=f"File: {filename}")
        self.lbl_score.config(text="Processing...", fg="#718093")
        self.root.update_idletasks()

        try:
            img = Image.open(filepath)

            # --- GUI Image Preview ---
            preview_img = img.copy()
            preview_img.thumbnail((350, 250))
            self.photo = ImageTk.PhotoImage(preview_img)
            self.drop_label.config(image=self.photo, text="")

            # --- API Preparation ---
            img_converted = img.convert('RGB')
            img_resized = img_converted.resize((224, 224))

            buffer = io.BytesIO()
            img_resized.save(buffer, format="JPEG")
            buffer.seek(0)

            files = {'file': (filename, buffer, 'image/jpeg')}

            response = requests.post(self.api_url, files=files, timeout=10)

            if response.status_code == 200:
                data = response.json()

                diagnosis = data["diagnosis"]
                confidence = data["confidence"]
                det = data["details"]

                self.lbl_details.config(
                    text=f"NORMAL: {det['NORMAL']:.1f}% | BACTERIA: {det['BACTERIA']:.1f}% | VIRUS: {det['VIRUS']:.1f}%"
                )
                self.lbl_score.config(text=f"{diagnosis} ({confidence:.1f}%)")

                if diagnosis == "NORMAL":
                    self.lbl_score.config(fg="#27ae60")
                elif diagnosis == "VIRUS":
                    self.lbl_score.config(fg="#e67e22")
                else:
                    self.lbl_score.config(fg="#c0392b")
            else:
                error_msg = response.json().get("detail", "Unknown server error")
                self.lbl_score.config(text=f"API Error: {error_msg}", fg="red")

        except requests.exceptions.ConnectionError:
            self.lbl_score.config(text="Connection Error: Is server running?", fg="red")
        except Exception as e:
            self.lbl_score.config(text=f"Error: {str(e)}", fg="red")


if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = ChestXrayClientApp(root)
    root.mainloop()