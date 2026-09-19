from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Get the main project folder path
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Folder for uploaded images
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")

# Automatically create the uploads folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# Load trained AI model
model = tf.keras.models.load_model(
    os.path.join(BASE_DIR, "models", "image_detector.keras")
)

IMG_SIZE = 128


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return "No image uploaded"

    file = request.files["image"]

    if file.filename == "":
        return "Please select an image"

    # Secure the uploaded filename
    filename = secure_filename(file.filename)

    # Save uploaded image
    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(filepath)

    # Open and preprocess image
    image = Image.open(filepath).convert("RGB")

    image = image.resize(
        (IMG_SIZE, IMG_SIZE)
    )

    img_array = np.array(image) / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # AI prediction
    prediction = model.predict(
        img_array
    )[0][0]

    # Determine result
    if prediction >= 0.5:

        result = "REAL IMAGE"
        confidence = prediction * 100

    else:

        result = "AI-GENERATED IMAGE"
        confidence = (1 - prediction) * 100

    return render_template(
        "result.html",
        result=result,
        confidence=round(float(confidence), 2),
        image_path="uploads/" + filename
    )


if __name__ == "__main__":

    app.run(
        debug=True
    )