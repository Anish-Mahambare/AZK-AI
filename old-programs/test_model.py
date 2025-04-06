import os
import torch
import onnx
import onnxruntime as ort
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

# ===========================
# 1. Load ONNX Model
# ===========================
onnx_path = 'melanoma_classifier.onnx'  # Path to the saved ONNX model
session = ort.InferenceSession(onnx_path)

# ===========================
# 2. Image Transformations
# ===========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Same normalization as before
])

# ===========================
# 3. Load Test Data (25 benign, 25 malignant)
# ===========================
test_dir = './melanoma_cancer_dataset/test'  # Adjust path if needed
benign_folder = os.path.join(test_dir, 'benign')
malignant_folder = os.path.join(test_dir, 'malignant')

# Load 25 benign and 25 malignant images
benign_images = [os.path.join(benign_folder, f) for f in os.listdir(benign_folder)][:25]
malignant_images = [os.path.join(malignant_folder, f) for f in os.listdir(malignant_folder)][:25]

# ===========================
# 4. Perform Inference on the Test Images
# ===========================
correct = 0
total = 0

for image_path in benign_images + malignant_images:
    # Open and preprocess image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).numpy()  # Add batch dimension and convert to numpy
    
    # Perform inference
    inputs = {session.get_inputs()[0].name: image}
    outputs = session.run(None, inputs)
    
    # Get predicted class (0 for benign, 1 for malignant)
    prediction = np.argmax(outputs[0], axis=1)[0]
    
    # Check if prediction is correct
    label = 0 if image_path in benign_images else 1
    if prediction == label:
        correct += 1
    total += 1

# ===========================
# 5. Calculate Accuracy
# ===========================
accuracy = (correct / total) * 100
print(f"Test Accuracy on 50 images: {accuracy:.2f}%")
