import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageTk
import numpy as np
import torchvision


# Paths to model weights
# Make sure these paths match where your weights actually are
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESNET_PATH = os.path.join(SCRIPT_DIR, 'full_modelres.pth')
CUSTOMCNN_PATH = os.path.join(SCRIPT_DIR, 'full_modelcnn.pth')

# Class names
CLASS_NAMES = ['fractured', 'not fractured']

import torch.nn as nn
import torch

class CustomCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Image preprocessing (must match training)
def get_infer_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def load_model(model_type):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        if model_type == 'ResNet50':
            if not os.path.exists(RESNET_PATH):
                raise FileNotFoundError(f"ResNet50 weights not found at {RESNET_PATH}")
                
            # Load the model (could be either a full model or just state_dict)
            loaded = torch.load(RESNET_PATH, map_location=device, weights_only=False)
            
            # Create the model architecture to match the saved checkpoint
            model = models.resnet50(weights=None)
            in_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Linear(in_features, 128),  # Changed from 256 to 128 to match checkpoint
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 2)  # Changed from 256 to 128 to match checkpoint
            )
            
            # If loaded object is a model, get its state_dict
            if isinstance(loaded, torch.nn.Module):
                model.load_state_dict(loaded.state_dict())
            else:  # It's already a state_dict
                model.load_state_dict(loaded)
        else:
            if not os.path.exists(CUSTOMCNN_PATH):
                raise FileNotFoundError(f"CustomCNN weights not found at {CUSTOMCNN_PATH}")
                
            # Load the model (could be either a full model or just state_dict)
            loaded = torch.load(CUSTOMCNN_PATH, map_location=device, weights_only=False)
            
            # Create the model
            model = CustomCNN(num_classes=2)
            
            # If loaded object is a model, get its state_dict
            if isinstance(loaded, torch.nn.Module):
                model.load_state_dict(loaded.state_dict())
            else:  # It's already a state_dict
                model.load_state_dict(loaded)
        model.eval()
        model.to(device)
        return model
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Failed to load model: {e}")

class FractureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Wrist Fracture Detection")
        self.root.geometry("420x500")
        self.model_type = tk.StringVar(value="ResNet50")
        self.model = load_model(self.model_type.get())
        self.image_path = None
        self.img_panel = None
        self.setup_ui()

    def setup_ui(self):
        # Configure root window
        self.root.minsize(500, 700)
        self.root.title("Wrist Fracture Detection")
        
        # Create main container with grid layout
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)  # Image area will expand
        
        # Model selection (row 0)
        model_frame = ttk.LabelFrame(main_frame, text="Select Model", padding=10)
        model_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        ttk.Radiobutton(model_frame, text="ResNet50", variable=self.model_type, 
                       value="ResNet50", command=self.reload_model).pack(side="left", padx=10)
        ttk.Radiobutton(model_frame, text="CustomCNN", variable=self.model_type, 
                       value="CustomCNN", command=self.reload_model).pack(side="left", padx=10)

        # Image display area (row 1)
        img_frame = ttk.LabelFrame(main_frame, text="X-ray Image", padding=5)
        img_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        
        # Canvas with scrollbar for image
        canvas = tk.Canvas(img_frame, bg='#f0f0f0')
        scrollbar = ttk.Scrollbar(img_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Image panel
        self.img_panel = ttk.Label(self.scrollable_frame)
        self.img_panel.pack(pady=10)

        # File selection area (row 2)
        self.file_frame = ttk.LabelFrame(main_frame, text="Select X-ray Image", padding=10)
        self.file_frame.grid(row=2, column=0, sticky="nsew", pady=(0, 10))
        
        # File selection button
        self.browse_btn = ttk.Button(
            self.file_frame,
            text="Browse Image Files",
            command=self.browse_image,
            width=20
        )
        self.browse_btn.pack(pady=20, padx=20, fill='x')
        
        # Label to show selected file name
        self.file_label = ttk.Label(
            self.file_frame,
            text="No file selected",
            wraplength=350,
            anchor="center",
            padding=10
        )
        self.file_label.pack(fill='x', padx=10, pady=(0, 10))
        
        # Button frame (row 3)
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        
        # Predict button - larger and more visible
        self.pred_btn = ttk.Button(
            btn_frame, 
            text="PREDICT FRACTURE", 
            command=self.predict,
            style="Accent.TButton"
        )
        self.pred_btn.pack(fill=tk.X, ipady=10)
        
        # Create a custom style for the button
        style = ttk.Style()
        style.configure(
            "Accent.TButton",
            font=('Helvetica', 12, 'bold'),
            padding=10,
            foreground='white',
            background='#0078d7'
        )
        
        # Output label (row 4)
        self.result_label = ttk.Label(
            main_frame, 
            text="", 
            font=("Arial", 14, "bold"),
            foreground="#333333",
            anchor="center",
            padding=10
        )
        self.result_label.grid(row=4, column=0, sticky="ew")
        
        # Configure grid weights for main frame rows
        main_frame.rowconfigure(1, weight=1)  # Image area expands

    def reload_model(self):
        self.model = load_model(self.model_type.get())
        self.result_label.config(text="")

    def browse_image(self):
        filetypes = [
            ("Image files", "*.png *.jpg *.jpeg *.bmp"),
            ("All files", "*.*")
        ]
        path = filedialog.askopenfilename(
            title="Select X-ray Image",
            filetypes=filetypes
        )
        if path and os.path.isfile(path):
            self.set_image(path)

    def set_image(self, path):
        try:
            self.image_path = path
            img = Image.open(path).convert('RGB')
            
            # Keep aspect ratio but limit the maximum size
            max_size = (400, 400)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Update image in the panel
            img_tk = ImageTk.PhotoImage(img)
            self.img_panel.configure(image=img_tk)
            self.img_panel.image = img_tk
            
            # Update the file info
            filename = os.path.basename(path)
            self.file_label.config(text=filename)
            self.result_label.config(text="")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
        
    def preprocess_image(self, image_path):
        """Preprocess the image for model prediction"""
        # Define the same transforms as during training
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        image = image.unsqueeze(0)  # Add batch dimension
        return image
        
    def predict(self):
        """Make a prediction on the loaded image"""
        if not hasattr(self, 'image_path') or not os.path.exists(self.image_path):
            messagebox.showerror("Error", "Please load an image first!")
            return
            
        if not hasattr(self, 'model') or self.model is None:
            messagebox.showerror("Error", "Please select a model first!")
            return
            
        try:
            # Disable predict button during prediction
            self.pred_btn.config(state=tk.DISABLED, text="Predicting...")
            self.root.update()
            
            # Preprocess the image
            image_tensor = self.preprocess_image(self.image_path)
            
            # Move to device (CPU/GPU)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            image_tensor = image_tensor.to(device)
            
            # Set model to evaluation mode and move to device
            self.model = self.model.to(device)
            self.model.eval()
            
            # Make prediction
            with torch.no_grad():
                output = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                confidence = confidence.item() * 100
                predicted_class = predicted.item()
                
            # Map model output to class names (adjust these indices if needed)
            # If results are flipped, swap the order of these class names
            class_names = ['Fracture', 'No Fracture']  # Index 0: Fracture, Index 1: No Fracture
            result = class_names[predicted_class]
            
            # Update UI with results
            color = "red" if result == "Fracture" else "green"
            self.result_label.config(
                text=f"Prediction: {result}\nConfidence: {confidence:.2f}%",
                foreground=color
            )
            
        except Exception as e:
            messagebox.showerror("Prediction Error", f"An error occurred: {str(e)}")
            print(f"Error during prediction: {str(e)}")
            
        finally:
            # Re-enable predict button
            self.pred_btn.config(state=tk.NORMAL, text="PREDICT FRACTURE")

if __name__ == "__main__":
    root = tk.Tk()
    app = FractureApp(root)
    root.mainloop()
