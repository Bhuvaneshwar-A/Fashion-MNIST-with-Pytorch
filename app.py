from flask import Flask, jsonify, request, render_template
import torch
import torchvision.transforms as transforms
from PIL import Image

app = Flask(__name__)

# Load the model
model= torch.load('models/MNIST_model_2.pth', map_location=torch.device('cpu'))


# Define the input image transformation
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Define the home page route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    file = request.files['image']

    # Open the image file and apply the transformation
    image = Image.open(file).convert('RGB')
    image = transform(image).unsqueeze(0)

    # Make a prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        result = predicted.item()

    # Return the prediction as a JSON response
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
