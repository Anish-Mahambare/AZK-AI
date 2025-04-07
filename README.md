# 🧬 Melanoma Classifier Web App

A simple Flask web application for classifying skin lesions as **Benign** or **Malignant** using a pre-trained ONNX model.

## 🚀 Features

- Upload an image of a skin lesion.
- Processes and classifies the image using a deep learning model in ONNX format.
- Returns the prediction (`Benign` or `Malignant`) via a web interface.
- Lightweight and easy to deploy locally.

## 🧠 Model Info

This app uses an ONNX model named `melanoma_classifier.onnx` trained for binary classification:
- **0** → Benign
- **1** → Malignant

The model expects input images resized to 224x224 and normalized accordingly.

## 🛠️ Requirements
Clone the repository to run the project. 

```bash
git clone https://github.com/Anish-Mahambare/AZK-AI.git
```
Then, navigate into the to the directory.

```bash
cd AZK-AI
```

Make sure you have the following Python packages installed:

```bash
pip install flask onnxruntime pillow torchvision numpy
```

## 📂 Project Structure

```
./
├── app.py               # Main Flask application
├── melanoma_classifier_50.onnx  # Pretrained ONNX model
├── old-programs/
│   └── xxxx       # python training files for model
├── templates/
│   └── index.html       # HTML template for file upload
└── README.md            # You're reading it!
```

## 🧪 How It Works

1. User accesses the homepage (`/`) and uploads an image.
2. The image is preprocessed using `torchvision` transforms:
   - Resize to 224x224
   - Convert to tensor and normalize
3. The image is passed to the ONNX model using `onnxruntime`.
4. A prediction is returned and displayed on the frontend as either:
   - ✅ Benign
   - ⚠️ Malignant

## 💻 Running the App

1. Make sure your environment is set up with the dependencies above.
2. Run the Flask app:

```bash
python app.py
```

3. Open your browser and go to [http://localhost:5000](http://localhost:5000)

## 🔒 Disclaimer

This tool is intended for **demonstration purposes only** and should **not be used for medical diagnosis**. Always consult with a healthcare professional.

