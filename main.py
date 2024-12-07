from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
SECRET_TOKEN = os.getenv("SECRET_TOKEN")

# Class names for the model
class_names = [
    "Aromatherapy Candle", "Balinese Topeng", "Barong T-shirt", "Beach Hat Bali", "Beach Sarong",
    "Beads Bracelet", "Bintang T-shirt", "Coconut Shell Candle Holder", "Crochet Bag", "Dream Catcher",
    "Hair Clip", "Handy Fan", "Keben (Balinese Woven Box)", "Keychain", "Rattan bag", "Silver Earrings",
    "Tridatu Bracelet", "Udeng (Balinese Headgear)", "Wall Decoration", "Wooden Earrings", "Woven Bag"
]

# Load the model
model = tf.keras.models.load_model('./model/model-souvenir-bali.h5')

# Create FastAPI app
app = FastAPI()

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
    None
        Raises an HTTPException if the token is invalid
    """
    if token != SECRET_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid authorization token")

@app.post("/predict")
async def predict(
    token: str = Form(...),
    image: UploadFile = File(...)
):
    """
    Handles image prediction requests by verifying the authorization token and processing the uploaded image file.
    
    This function expects a POST request with a form-data payload containing an 'image' file and a 'token'.
    It verifies the token against a pre-defined SECRET_TOKEN and processes the image for prediction using a 
    pre-loaded deep learning model. If successful, it returns the predicted class and accuracy. Otherwise, 
    it returns an appropriate error message.
    
    Returns
    -------
    JSONResponse
        A JSON response containing the prediction result if successful, or an error message if not.
    """
    # Verify the token
    verify_token(token)

    # Validate and process the image
    try:
        content = await image.read()
        img = Image.open(io.BytesIO(content))
        img_array = prepare_image(img)

        # Make predictions
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions)
        predicted_class = class_names[predicted_class_index]
        accuracy = float(predictions[0][predicted_class_index])

        return JSONResponse(content={
            "success": "true",
            "message": "Model is predicted successfully",
            "data": {
                "result": predicted_class,
                "accuration": round(accuracy, 4),
            }
        })
    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse(content={
            "success": "false",
            "message": "An error occurred while making the prediction."
        }, status_code=500)
