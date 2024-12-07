from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from dotenv import load_dotenv

load_dotenv()

SECRET_TOKEN = os.getenv("SECRET_TOKEN")
print(SECRET_TOKEN)

class_names = [
    "Aromatherapy Candle", "Balinese Topeng", "Barong T-shirt", "Beach Hat Bali", "Beach Sarong",
    "Beads Bracelet", "Bintang T-shirt", "Coconut Shell Candle Holder", "Crochet Bag", "Dream Catcher",
    "Hair Clip", "Handy Fan", "Keben (Balinese Woven Box)", "Keychain", "Rattan bag", "Silver Earrings",
    "Tridatu Bracelet", "Udeng (Balinese Headgear)", "Wall Decoration", "Wooden Earrings", "Woven Bag"
]

model = tf.keras.models.load_model('model-surevenir.h5')
print("Model loaded with custom objects")

app = Flask(__name__)

def prepare_image(img):
    """
    Prepares an image for inference by resizing it to the model's required size (224x224) and normalizing its pixel values to be between 0 and 1.
    
    Parameters
    ----------
    img : PIL.Image
        The image to be prepared
    
    Returns
    -------
    np.ndarray
        A 3D numpy array of shape (1, 224, 224, 3) representing the prepared image
    """
    img = img.resize((224, 224))  
    img_array = np.array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  
    return img_array

def verify_token(token):
    """
    Verifies that the given token matches the SECRET_TOKEN.
    
    Parameters
    ----------
    token : str
        The token to verify
    
    Returns
    -------
    bool
        True if the token is valid, False if not
    """
    print(token)
    if token != SECRET_TOKEN:
        return False
    return True

@app.route('/predict', methods=['POST'])
def predict():
    
    """
    Handles image prediction requests by verifying the authorization token and processing the uploaded image file.
    
    This function expects a POST request with a form-data payload containing an 'image' file and a 'token'.
    It verifies the token against a pre-defined SECRET_TOKEN and processes the image for prediction using a 
    pre-loaded deep learning model. If successful, it returns the predicted class and accuracy. Otherwise, 
    it returns an appropriate error message.
    
    Returns
    -------
    flask.Response
        A JSON response containing the prediction result if successful, or an error message if not.
    """
    token = request.form.get('token')

    
    if not token:
        return jsonify({
            "success": "false", 
            "message": "No authorization token"
        })
    
    if not verify_token(token):
        return jsonify({
            "success": "false", 
            "message": "Invalid authorization token"
        })
    
    if 'image' not in request.files:
        return jsonify({
            "success": "false", 
            "message": "No file part"
        })
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({
            "success": "false", 
            "message": "No selected image"
        })
    
    try:
        
        img = Image.open(io.BytesIO(file.read()))
        
        
        img_array = prepare_image(img)
        
        
        predictions = model.predict(img_array)
        
        
        predicted_class_index = np.argmax(predictions)
        predicted_class = class_names[predicted_class_index]
        accuracy = float(predictions[0][predicted_class_index])  
        
        return jsonify({
            "success": "true",
            "message": "Model is predicted successfully",
            "data": {
                "result": predicted_class,
                "accuration": round(accuracy, 4),
            }
        })
    
    except Exception as e:
        print(f"Error: {e}")  
        return jsonify({
            "success": "false",
            "message": "An error occurred while making the prediction."
        })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
