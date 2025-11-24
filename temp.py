import customtkinter as ctk
import cv2
import torch
import threading
import time
import io
import pygame
import string
import numpy as np
from gtts import gTTS
from PIL import Image

# --- DEPENDENCY CHECK ---
try:
    from transformers import AutoModelForImageClassification, AutoImageProcessor
except ImportError:
    print("\nCRITICAL ERROR: 'transformers' library is missing.")
    print("Please run: pip install transformers")
    exit()

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
# You can use local paths (folders containing model.safetensors & config.json)
# OR you can use the HuggingFace ID strings directly if you have internet.

# OPTION A: If you downloaded files to folders:
ASL_MODEL_PATH = "C:/Users/Ahas Kaushik/Desktop/models/asl_model_folder" 
ISL_MODEL_PATH = "C:/Users/Ahas Kaushik/Desktop/models/isl_model_folder"

# OPTION B: If you want to download/run directly from internet (Easiest):
# ASL_MODEL_PATH = "prithivMLmods/Alphabet-Sign-Language-Detection"
# ISL_MODEL_PATH = "Hemg/Indian-sign-language-classification"

# ==============================================================================
# 2. HELPER CLASSES
# ==============================================================================

class TextToSpeech:
    def __init__(self):
        pygame.mixer.init()
        self.lock = threading.Lock()

    def speak(self, text):
        threading.Thread(target=self._speak_thread, args=(text,), daemon=True).start()

    def _speak_thread(self, text):
        with self.lock:
            if pygame.mixer.music.get_busy():
                return
            try:
                mp3_fp = io.BytesIO()
                tts = gTTS(text=text, lang='en')
                tts.write_to_fp(mp3_fp)
                mp3_fp.seek(0)
                pygame.mixer.music.load(mp3_fp)
                pygame.mixer.music.play()
            except Exception as e:
                print(f"TTS Error: {e}")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Universal Sign Language Recognizer")
        self.geometry("1100x700")
        
        self.mode = "ASL"
        self.running = False
        self.cap = None
        
        self.model = None
        self.processor = None
        self.id2label = {}
        
        self.last_prediction = ""
        self.tts = TextToSpeech()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.setup_ui()
        
        # Initial Load
        self.load_model_and_start(ASL_MODEL_PATH, "ASL")

    def setup_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Left Panel ---
        self.left_frame = ctk.CTkFrame(self, width=300, corner_radius=10)
        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.lbl_mode = ctk.CTkLabel(self.left_frame, text="Current Mode: ASL", font=("Arial", 20, "bold"))
        self.lbl_mode.pack(pady=20)

        self.btn_switch = ctk.CTkButton(
            self.left_frame, text="Switch to ISL", command=self.toggle_mode,
            height=50, fg_color="#e74c3c", font=("Arial", 16)
        )
        self.btn_switch.pack(pady=10, padx=20, fill="x")

        self.lbl_log = ctk.CTkLabel(self.left_frame, text="Detection Log:", anchor="w")
        self.lbl_log.pack(pady=(20, 5), padx=20, fill="x")
        self.textbox = ctk.CTkTextbox(self.left_frame)
        self.textbox.pack(pady=10, padx=20, fill="both", expand=True)

        # --- Right Panel ---
        self.right_frame = ctk.CTkFrame(self)
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.lbl_video = ctk.CTkLabel(self.right_frame, text="Initializing...", text_color="gray")
        self.lbl_video.pack(expand=True, fill="both")

    def toggle_mode(self):
        self.stop_feed()
        self.lbl_video.configure(text="Switching Models...", image=None)
        self.update()

        if self.mode == "ASL":
            self.mode = "ISL"
            target_path = ISL_MODEL_PATH
            self.btn_switch.configure(text="Switch to ASL", fg_color="#3498db")
            self.lbl_mode.configure(text="Current Mode: ISL")
        else:
            self.mode = "ASL"
            target_path = ASL_MODEL_PATH
            self.btn_switch.configure(text="Switch to ISL", fg_color="#e74c3c")
            self.lbl_mode.configure(text="Current Mode: ASL")

        self.log(f"System: Switched to {self.mode}")
        self.load_model_and_start(target_path, self.mode)

    def load_model_and_start(self, path, mode_name):
        try:
            print(f">> Loading Model from: {path}")
            
            # --- THE MAGIC PART ---
            # AutoImageProcessor: Automatically handles resizing/normalization for specific model
            self.processor = AutoImageProcessor.from_pretrained(path)
            
            # AutoModelForImageClassification: Automatically detects architecture (ViT, ResNet, etc.)
            self.model = AutoModelForImageClassification.from_pretrained(path)
            
            self.model.to(self.device)
            self.model.eval()
            
            # Extract Labels directly from the model config!
            self.id2label = self.model.config.id2label
            
            self.log(f"System: Loaded {mode_name} (Architecture: {self.model.config.architectures[0]})")
            self.start_feed()
                
        except Exception as e:
            self.log(f"Error loading model: {e}")
            print(f"Detailed Error: {e}")
            # We start feed anyway so you can see camera
            self.start_feed()

    def start_feed(self):
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.video_loop()

    def stop_feed(self):
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def video_loop(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            
            # Convert BGR (OpenCV) to RGB (PIL)
            # Hugging Face Processors expect PIL Images or RGB Arrays
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            detected_text = None
            
            if self.model and self.processor:
                try:
                    # 1. Preprocess using the model's specific processor
                    inputs = self.processor(images=pil_image, return_tensors="pt")
                    inputs = inputs.to(self.device)
                    
                    # 2. Inference
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        logits = outputs.logits
                        probs = torch.nn.functional.softmax(logits, dim=-1)
                        confidence, pred_idx = torch.max(probs, dim=-1)
                        
                        # 3. Get Label
                        if confidence.item() > 0.5: # 50% confidence threshold
                            label_idx = pred_idx.item()
                            # Fetch label string from model config
                            detected_text = self.id2label.get(label_idx, str(label_idx))
                            
                except Exception as e:
                    print(f"Inference Error: {e}")

            if detected_text:
                # Clean up label (sometimes they look like "LABEL_0" or "0")
                if detected_text.startswith("LABEL_"):
                    detected_text = detected_text.replace("LABEL_", "")
                
                # Draw on screen
                cv2.putText(frame, f"{self.mode}: {detected_text} ({int(confidence.item()*100)}%)", (30, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if detected_text != self.last_prediction:
                    self.last_prediction = detected_text
                    self.log(f"Detected: {detected_text}")
                    self.tts.speak(detected_text)

            # Convert to CTk Image
            ctk_img = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(640, 480))
            
            self.lbl_video.configure(image=ctk_img)
            self.lbl_video.image = ctk_img

        self.after(10, self.video_loop)

    def log(self, message):
        self.textbox.insert("end", message + "\n")
        self.textbox.see("end")

    def on_close(self):
        self.stop_feed()
        self.destroy()

if __name__ == "__main__":
    ctk.set_appearance_mode("Dark")
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()