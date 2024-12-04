from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from dotenv import load_dotenv

load_dotenv()

SECRET_TOKEN = os.getenv("SECRET_TOKEN")

class_names = [
    "Aromatherapy Candle", "Balinese Topeng", "Barong T-shirt", "Beach Hat Bali", "Beach Sarong",
    "Beads Bracelet", "Bintang T-shirt", "Coconut Shell Candle Holder", "Crochet Bag", "Dream Catcher",
    "Hair Clip", "Handy Fan", "Keben (Balinese Woven Box)", "Keychain", "Rattan bag", "Silver Earrings",
    "Tridatu Bracelet", "Udeng (Balinese Headgear)", "Wall Decoration", "Wooden Earrings", "Woven Bag"
]

model = tf.keras.models.load_model('./model/model.keras')

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

if __name__ == '__main__':
    app.run(debug=True)