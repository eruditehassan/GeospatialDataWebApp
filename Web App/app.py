from flask import Flask, request, render_template, jsonify
import os
import uuid
from werkzeug.utils import secure_filename
from PIL import Image
import mimetypes

import json
import cv2
import numpy as np
import base64

import geopandas as gpd
from zipfile import ZipFile
from tensorflow.keras.models import load_model
import boto3

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allowed extensions for image files
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}

lambda_client = boto3.client('lambda', region_name='eu-west-2')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_image_mime_type(filename):
    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type and mime_type.startswith('image')

def clear_uploads_folder():
    for root, dirs, files in os.walk(UPLOAD_FOLDER, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Generate a unique directory for each upload
    unique_folder = os.path.join(UPLOAD_FOLDER, str(uuid.uuid4()))
    os.makedirs(unique_folder, exist_ok=True)

    filepath = os.path.join(unique_folder, file.filename)
    file.save(filepath)

    if filepath.endswith('.zip'):
        # Handle ZIP file containing shapefiles
        with ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(unique_folder)
            extracted_files = zip_ref.namelist()
        
        # Find the .shp file among extracted files
        shp_file = next((f for f in extracted_files if f.endswith('.shp')), None)
        if shp_file:
            # Construct full path for the .shp file
            shp_filepath = os.path.join(unique_folder, shp_file)
            # Ensure all associated files are in the same directory
            gdf = gpd.read_file(shp_filepath)
            geojson_data = gdf.to_json()
            return jsonify({'geojson': geojson_data})
        else:
            return jsonify({'error': 'No valid shapefile found in the ZIP archive'})
    else:
        # Handle regular GeoJSON files
        try:
            gdf = gpd.read_file(filepath)
            geojson_data = gdf.to_json()
            return jsonify({'geojson': geojson_data})
        except Exception as e:
            return jsonify({'error': str(e)})

@app.route('/predict_image', methods=['POST'])
def predict_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'})

    image = request.files['image']
    if image.filename == '':
        return jsonify({'error': 'No selected image'})

    # Check if the file is an allowed image type
    if not allowed_file(image.filename):
        return jsonify({'error': 'Invalid file type. Please upload an image file (jpg, jpeg, png, gif).'})

    # Check MIME type
    if not is_image_mime_type(image.filename):
        return jsonify({'error': 'Invalid image MIME type. Please upload a valid image file.'})

    # Save the image file
    image_filename = secure_filename(image.filename)
    image_path = os.path.join(UPLOAD_FOLDER, image_filename)
    image.save(image_path)

    with open(image_path, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    event = {
        'image_data': encoded_image
    }

    # Invoke the Lambda function
    response = lambda_client.invoke(
        FunctionName='ml_preprocessing',  
        InvocationType='RequestResponse',
        Payload=json.dumps(event)
    )

    # Parse the response
    response_payload = response['Payload'].read().decode('utf-8')
    result = json.loads(response_payload)

    body_str = result['body']
    body_dict = json.loads(body_str)
    processed_image = np.array(body_dict['processed_image'])

    loaded_model = load_model('models/ResNet50_eurosat.h5')

    predictions = loaded_model.predict(processed_image)

    predicted_class_index = np.argmax(predictions, axis=1)

    class_names = ['AnnualCrop','Forest','HerbaceousVegetation','Highway','Industrial','Pasture','PermanentCrop','Residential','River','SeaLake']

    predicted_class_name = class_names[predicted_class_index[0]]
    
    return jsonify({'prediction': 'The prediction is: ' + predicted_class_name})

def preprocess_image(image_data, target_size=(64, 64)):
    # Decode the base64 image
    image_array = np.frombuffer(base64.b64decode(image_data), np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    # Check if the image was loaded correctly
    if image is None:
        raise ValueError("The provided image data could not be decoded.")
    
    # Resize the image to the target size
    image = cv2.resize(image, target_size)
    
    # Convert the image from BGR to RGB (OpenCV loads images in BGR format)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalize the image (assuming the model expects values in the range [0, 1])
    image = image.astype(np.float32) / 255.0
    
    # Expand dimensions to match the model's input shape (batch size, height, width, channels)
    image = np.expand_dims(image, axis=0)
    
    return image

def lambda_handler(event, context):
    try:
        # Assuming the image is sent as a base64-encoded string in the JSON event
        image_data = event['image_data']
        
        # Preprocess the image
        processed_image = preprocess_image(image_data)
        
        # Convert processed image back to list for JSON serialization
        processed_image_list = processed_image.tolist()
        
        # Return the processed image data
        return {
            'statusCode': 200,
            'body': json.dumps({'processed_image': processed_image_list})
        }
    
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

if __name__ == '__main__':
    clear_uploads_folder()
    #app.run(debug=True)
    app.run(host='0.0.0.0', port=80)
    
