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

if __name__ == '__main__':
    app.run(debug=True)