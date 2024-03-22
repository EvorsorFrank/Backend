from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Form, Path, HTTPException
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_BEANS = tf.keras.models.load_model("../saved_models/Bean_CNN_Model_V2.keras")
MODEL_CORN = tf.keras.models.load_model("../saved_models/Corn_CNN_Model_V2.keras")
MODEL = tf.keras.models.load_model("../saved_models/Tomato_CNN_Model_V2.keras")
MODEL_RICE = tf.keras.models.load_model("../saved_models/Rice_CNN_Model_V1.keras")


BEANS_CLASS_NAMES = [
    'Bean Angular Leaf Spot',
    'Healthy Bean',
    'Bean Rust',
]

CORN_CLASS_NAMES = [
    'Corn Common Rust',
    'Corn Gray Leaf Spot',
    'Healthy Corn',
    'Corn Northern Leaf Blight'
]

RICE_CLASS_NAMES = [
    'Rice Bacterial Leaf Blight',
    'Rice Brown Spot',
    'Healthy Rice',
    'Rice Leaf Blast',
    'Rice Leaf Scald',
    'Rice Narrow Brown Spot'
 ]

TOMATO_CLASS_NAMES = [
    'Tomato Bacterial Spot',
    'Tomato Early Blight',
    'Healthy Tomato',
    'Tomato Late Blight',
    'Tomato Leaf Mold',
    'Tomato Mosaic Virus',
    'Tomato Septoria Leaf Spot',
    'Tomato Target Spot',
    'Tomato Yellow Leaf Curl Virus'
]


def read_file_as_image(data) -> np.ndarray:
    image_size = (256, 256)
    im = Image.open(BytesIO(data))
    # Resize the image to the desired dimensions and convert it to RGB mode
    resized = im.resize(image_size).convert("RGB")
    # Convert the image to a numpy array
    image = np.array(resized)
    return image


@app.get('/')
def hello_world():
    return 'Hello World'


@app.get('/hello')
def hello_cow():
    return 'Hello betch'


@app.post('/upload_image')
async def upload_image(
    file: UploadFile = File(...)
):
    image = read_file_as_image(file.read())
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(img_batch)
    predicted_class = TOMATO_CLASS_NAMES[np.argmax(predictions)]
    print(predicted_class)
    print(predictions)

    return {"class": predicted_class}


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
