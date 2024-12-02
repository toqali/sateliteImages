import os
from flask import Flask, request, render_template, send_from_directory
import torch
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread

app = Flask(__name__)

# Folder to store uploaded images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the uploads folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the trained model
MODEL_PATH = 'Unet_res.pth'
Unet_res = torch.load(MODEL_PATH, map_location=torch.device('cpu'))

# Image normalization function
def normalization(img):
    for c in range(12):  # Assuming 12 channels
        min_val = np.min(img[:, :, c])
        max_val = np.max(img[:, :, c])
        if max_val > min_val:
            img[:, :, c] = (img[:, :, c] - min_val) / (max_val - min_val)
        else:
            img[:, :, c] = 0  # Set to 0 if no variation

# Image preprocessing function (resize and convert to tensor)
def preprocess_image(image):
    # Normalize image
    normalization(image)
    
    # Add batch dimension and transpose to PyTorch format
    image = np.expand_dims(image, axis=0)  # (1, 128, 128, 12)
    image = np.transpose(image, (0, 3, 1, 2))  # (1, 12, 128, 128)
    
    # Convert to torch tensor and move to device
    image_tensor = torch.from_numpy(image.astype(np.float32)).to('cuda' if torch.cuda.is_available() else 'cpu')
    return image_tensor

# Ensure values are in 0..1 range for display
def normalize_for_display(image):
    image = image.astype(np.float32)
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val > min_val:
        image = (image - min_val) / (max_val - min_val)
    # Clip values to ensure they are in the correct range
    return np.clip(image, 0, 1)

# Route for uploading and displaying image
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        
        file = request.files['file']

        if file.filename == '':
            return 'No selected file'

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            image = imread(filepath)

            # Normalize and save the original image (first 3 channels) for display
            display_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.png')
            if image.shape[-1] >= 3:
                display_image = normalize_for_display(image[:, :, :3])  # Normalize first 3 channels for RGB-like display
                plt.imsave(display_image_path, display_image)
            else:
                display_image = normalize_for_display(image[:, :, 0])  # Normalize first channel for grayscale display
                plt.imsave(display_image_path, display_image, cmap='gray')

            # Preprocess the image for model input
            input_tensor = preprocess_image(image)

            # Run the image through the model to get predictions
            with torch.no_grad():
                prediction = Unet_res(input_tensor)

            # Post-process prediction (convert to NumPy and squeeze)
            prediction = prediction.cpu().detach().numpy()
            prediction = np.squeeze(prediction, axis=0)  # Remove batch dimension
            prediction = np.transpose(prediction, (1, 2, 0))  # Convert to (128, 128, channels)

            # Normalize prediction for display (since it's probably between -1 and 1 or some other range)
            prediction = normalize_for_display(prediction[:, :, 0])

            # Save the prediction as an image to display
            prediction_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'prediction.png')
            plt.imsave(prediction_image_path, prediction, cmap='gray')  # Assuming it's a 1-channel output

            # Return the page with uploaded and predicted images
            return render_template('display_image.html', 
                                   uploaded_filename='uploaded_image.png',
                                   prediction_filename='prediction.png')

    return render_template('upload_image.html')

# Route to serve the uploaded image
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
