import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.models import ResNet18_Weights

BASE_DIR = './melanoma_cancer_dataset'  # Update if needed
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TEST_DIR = os.path.join(BASE_DIR, 'test')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print(f"Classes: {train_dataset.classes}")  # ['benign', 'malignant']

def show_batch(loader, dataset):
    images, labels = next(iter(loader))
    plt.figure(figsize=(10, 8))
    for i in range(8):
        img = images[i].permute(1, 2, 0)
        label = dataset.classes[labels[i]]
        plt.subplot(2, 4, i + 1)
        plt.imshow(img.numpy() * 0.5 + 0.5)
        plt.title(label)
        plt.axis('off')
    plt.show()


model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1) 
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 50
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(train_loader):.4f}")

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Test Accuracy: {100 * correct / total:.2f}%")

from PIL import Image

def predict_image(img_path, model, transform, class_names):
    model.eval()
    image = Image.open(img_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_class = class_names[predicted.item()]
        print(f"Predicted class: {predicted_class}")


test_image_path = './melanoma_cancer_dataset/test/benign/melanoma_10000.jpg'  
predict_image(test_image_path, model, transform, train_dataset.classes)

torch.save(model.state_dict(), 'melanoma_classifier_50.pth')

dummy_input = torch.randn(1, 3, 224, 224).to(device)
onnx_path = "melanoma_classifier_50.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=11,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)
print(f"ONNX model saved to: {onnx_path}")

