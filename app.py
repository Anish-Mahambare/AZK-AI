from flask import Flask, render_template, request, jsonify
import onnxruntime as ort
from PIL import Image
import numpy as np
import io
from torchvision import transforms

app = Flask(__name__)


onnx_path = 'melanoma_classifier_50.onnx'
session = ort.InferenceSession(onnx_path)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  
])


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400


    image = Image.open(file.stream).convert('RGB')
    image = transform(image).unsqueeze(0).numpy()  

  
    inputs = {session.get_inputs()[0].name: image}
    outputs = session.run(None, inputs)
    prediction = np.argmax(outputs[0], axis=1)[0]

   
    if prediction == 0:
        result = 'Benign'
    else:
        result = 'Malignant'

    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
