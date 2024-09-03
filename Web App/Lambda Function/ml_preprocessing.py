import json
import cv2
import numpy as np
import base64
import boto3

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