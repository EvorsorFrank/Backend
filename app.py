from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import tensorflow as tf
from PIL import Image
import numpy as np
from geopy.geocoders import Nominatim
from collections import defaultdict


app = Flask(__name__)
CORS(app)
#app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://capstone_samonte_db_user:j3vFR3PT7BepaBcEl789ocCL5UzzsYfi@dpg-cnqqb5v109ks73fd77pg-a.singapore-postgres.render.com/capstone_samonte_db'
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:admin123@localhost:5432/test-capstone'

db = SQLAlchemy(app)


class Task(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    title = db.Column(db.String(200), nullable=False)
    done = db.Column(db.Boolean, default=False)


with app.app_context():
    db.create_all()


@app.route('/')
def index():
    return "hello"


@app.route('/tasks')
def get_tasks():
    tasks = Task.query.all()
    task_list = [
        {'id': task.id, 'title': task.title, 'done': task.done} for task in tasks
    ]
    return jsonify({"tasks": task_list})


def count_identifications():
    tomato_count = TomatoData.query.count()
    corn_count = CornData.query.count()
    rice_count = RiceData.query.count()
    beans_count = BeansData.query.count()
    return {
        'Tomato': tomato_count,
        'Corn': corn_count,
        'Rice': rice_count,
        'Beans': beans_count
    }


@app.route('/identification_count', methods=['GET'])
def get_identification_count():
    counts = count_identifications()
    print(counts)
    return jsonify(counts)


class TomatoData(db.Model):
    __tablename__ = 'tomato_data'
    data_id = db.Column(db.Integer, primary_key=True)
    identification_type = db.Column(db.String(64))
    plant_disease = db.Column(db.String(64))
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    address = db.Column(db.String(256))
    city = db.Column(db.String(64))
    date_recorded = db.Column(db.Date)

class CornData(db.Model):
    __tablename__ = 'corn_data'
    data_id = db.Column(db.Integer, primary_key=True)
    identification_type = db.Column(db.String(64))
    plant_disease = db.Column(db.String(64))
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    address = db.Column(db.String(256))
    city = db.Column(db.String(64))
    date_recorded = db.Column(db.Date)

class RiceData(db.Model):
    __tablename__ = 'rice_data'
    data_id = db.Column(db.Integer, primary_key=True)
    identification_type = db.Column(db.String(64))
    plant_disease = db.Column(db.String(64))
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    address = db.Column(db.String(256))
    city = db.Column(db.String(64))
    date_recorded = db.Column(db.Date)

class BeansData(db.Model):
    __tablename__ = 'beans_data'
    data_id = db.Column(db.Integer, primary_key=True)
    identification_type = db.Column(db.String(64))
    plant_disease = db.Column(db.String(64))
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    address = db.Column(db.String(256))
    city = db.Column(db.String(64))
    date_recorded = db.Column(db.Date)

with app.app_context():
    db.create_all()


def create_model_class(predict_type):   
    class_name = predict_type + 'Data'
    attributes = {
        'data_id': db.Column(db.Integer, primary_key=True),
        'identification_type': db.Column(db.String(64)),
        'plant_disease': db.Column(db.String(64)),
        'latitude': db.Column(db.Float),
        'longitude': db.Column(db.Float),
        'address': db.Column(db.String(256)),
        'date_recorded': db.Column(db.Date)
    }
    # Check if the class already exists
    if class_name not in globals():
        model_class = type(class_name, (db.Model,), attributes)
    else:
        # If the class already exists, extend the existing one
        model_class = type(class_name, (globals()[class_name], db.Model), attributes)
    return model_class

def predict_models(predict_type):
    # Define the models and class names
    models = {
        'Beans': tf.keras.models.load_model("./saved_models/Bean_CNN_Model_V2.keras"),
        'Corn': tf.keras.models.load_model("./saved_models/Corn_CNN_Model_V2.keras"),
        'Rice': tf.keras.models.load_model("./saved_models/Rice_CNN_Model_V1.keras"),
        'Tomato': tf.keras.models.load_model("./saved_models/Tomato_CNN_Model_V2.keras")
    }

    class_names = {
        'Beans': ['Bean Angular Leaf Spot', 'Healthy Bean', 'Bean Rust'],
        'Corn': ['Corn Common Rust', 'Corn Gray Leaf Spot', 'Healthy Corn', 'Corn Northern Leaf Blight'],
        'Rice': ['Rice Bacterial Leaf Blight', 'Rice Brown Spot', 'Healthy Rice', 'Rice Leaf Blast', 'Rice Leaf Scald', 'Rice Narrow Brown Spot'],
        'Tomato': ['Tomato Bacterial Spot', 'Tomato Early Blight', 'Healthy Tomato', 'Tomato Late Blight', 'Tomato Leaf Mold', 'Tomato Mosaic Virus', 'Tomato Septoria Leaf Spot', 'Tomato Target Spot', 'Tomato Yellow Leaf Curl Virus']
    }

    # Get the appropriate model and class names based on predictType
    model_type = models.get(predict_type)
    if model_type is None:
        return None, None  # Handle invalid predictType

    return model_type, class_names[predict_type]


def predict_models(predict_type):
    # Define the models and class names
    models = {
        'Beans': tf.keras.models.load_model("./saved_models/Bean_CNN_Model_V2.keras"),
        'Corn': tf.keras.models.load_model("./saved_models/Corn_CNN_Model_V2.keras"),
        'Rice':  tf.keras.models.load_model("./saved_models/Rice_CNN_Model_V1.keras"),
        'Tomato': tf.keras.models.load_model("./saved_models/Tomato_CNN_Model_V2.keras"),
    }

    class_names = {
        'Beans': [
            'Bean Angular Leaf Spot',
            'Healthy Bean',
            'Bean Rust',
        ],
        'Corn': [
            'Corn Common Rust',
            'Corn Gray Leaf Spot',
            'Healthy Corn',
            'Corn Northern Leaf Blight'
        ],
        'Rice': [
            'Rice Bacterial Leaf Blight',
            'Rice Brown Spot',
            'Healthy Rice',
            'Rice Leaf Blast',
            'Rice Leaf Scald',
            'Rice Narrow Brown Spot'
        ],
        'Tomato': [
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
    }
    # Get the appropriate model and class names based on predictType
    model_type = models.get(predict_type)
    if model_type is None:
        return None, None  # Handle invalid predictType

    return model_type, class_names[predict_type]


def read_file_as_image(data) -> np.ndarray:
    image_size = (256, 256)
    im = Image.open(data)
    # Resize the image to the desired dimensions and convert it to RGB mode
    resized = im.resize(image_size).convert("RGB")
    # Convert the image to a numpy array
    image = np.array(resized)
    return image


def reverse_geocode(latitude, longitude):
    # Initialize the Nominatim geocoder
    geolocator = Nominatim(user_agent="geocodingMPPDI")

    # Make a reverse geocoding request
    location = geolocator.reverse((latitude, longitude), language="en")

    # Extract and return the address
    return location.address if location else "Location not found"


def reverse_geocode_city(latitude, longitude):
    # Initialize the Nominatim geocoder
    geolocator = Nominatim(user_agent="geocodingMPPDI")

    # Make a reverse geocoding request
    location = geolocator.reverse((latitude, longitude), language="en")

    # Extract and return the city
    if location:
        address_components = location.raw.get("address", {})
        print("Address Components:", address_components)  # Print statement
        city = address_components.get("town", "")
        if city:
            return city
        else:
            city = address_components.get("city", "")
            if city:
                return city
        county = address_components.get("county", "")
        if county:
            return county
    else:
        return ""


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    predict_type = request.form['predictType']
    latitude = request.form['latitude']
    longitude = request.form['longitude']
    date = request.form['date']
    # Convert latitude and longitude to float
    latitude_float = float(latitude)
    longitude_float = float(longitude)
    city = reverse_geocode_city(latitude_float, longitude_float)
    address = reverse_geocode(latitude, longitude)
    model_type, class_names = predict_models(predict_type)
    image = read_file_as_image(file)
    img_batch = np.expand_dims(image, 0)
    prediction = model_type.predict(img_batch)
    predicted_class = class_names[np.argmax(prediction)]
    print(prediction)
    # Check if the predicted disease is "Healthy"
    if predicted_class.startswith("Healthy"):
        return jsonify({"class": predicted_class}), 200
    else:
        # Instantiate the appropriate class based on predict_type
        if predict_type == 'Tomato':
            new_record = TomatoData(
                identification_type=predict_type,
                plant_disease=predicted_class,
                latitude=latitude_float,
                longitude=longitude_float,
                address=address,
                city=city,
                date_recorded=date
            )
        elif predict_type == 'Corn':
            new_record = CornData(
                identification_type=predict_type,
                plant_disease=predicted_class,
                latitude=latitude_float,
                longitude=longitude_float,
                address=address,
                city=city,
                date_recorded=date
            )
        elif predict_type == 'Rice':
            new_record = RiceData(
                identification_type=predict_type,
                plant_disease=predicted_class,
                latitude=latitude_float,
                longitude=longitude_float,
                address=address,
                city=city,
                date_recorded=date
            )
        elif predict_type == 'Beans':
            new_record = BeansData(
                identification_type=predict_type,
                plant_disease=predicted_class,
                latitude=latitude_float,
                longitude=longitude_float,
                address=address,
                city=city,
                date_recorded=date
            )

        # Add the new record to the database
        db.session.add(new_record)
        db.session.commit()

    return jsonify({"class": predicted_class}), 200



@app.route('/plant_disease_counts', methods=['GET'])
def get_plant_disease_counts_by_city():
    # Get all records for each plant type
    tomato_records = TomatoData.query.all()
    corn_records = CornData.query.all()
    rice_records = RiceData.query.all()
    beans_records = BeansData.query.all()

    # Initialize a defaultdict to store plant disease counts by city
    city_counts = defaultdict(lambda: defaultdict(int))

    # Function to update city_counts with records of a specific plant type
    def update_counts(records, plant_type):
        for record in records:
            # Extract city directly from the record
            city = record.city if record.city else "Unknown"
            city_counts[city][plant_type] += 1

    # Update counts for each plant type
    update_counts(tomato_records, 'Tomato')
    update_counts(corn_records, 'Corn')
    update_counts(rice_records, 'Rice')
    update_counts(beans_records, 'Beans')

    # Calculate total counts for each city
    city_totals = {}
    for city, counts in city_counts.items():
        total_count = sum(counts.values())
        city_totals[city] = total_count

    return jsonify(city_totals)

@app.route('/plant_disease_rankings', methods=['GET'])
def get_plant_disease_rankings():
    # Get all records for each plant type
    tomato_records = TomatoData.query.all()
    corn_records = CornData.query.all()
    rice_records = RiceData.query.all()
    beans_records = BeansData.query.all()

    # Initialize defaultdicts to store plant disease counts for each crop
    tomato_counts = defaultdict(int)
    corn_counts = defaultdict(int)
    rice_counts = defaultdict(int)
    beans_counts = defaultdict(int)

    # Function to update counts for each crop
    def update_counts(records, counts):
        for record in records:
            counts[record.plant_disease] += 1

    # Update counts for each crop
    update_counts(tomato_records, tomato_counts)
    update_counts(corn_records, corn_counts)
    update_counts(rice_records, rice_counts)
    update_counts(beans_records, beans_counts)

    # Sort the diseases by count in descending order for each crop
    sorted_tomato_diseases = sorted(tomato_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_corn_diseases = sorted(corn_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_rice_diseases = sorted(rice_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_beans_diseases = sorted(beans_counts.items(), key=lambda x: x[1], reverse=True)

    # Create a list of dictionaries containing disease rankings for each crop
    tomato_rankings = [{'disease': disease, 'count': count} for disease, count in sorted_tomato_diseases]
    corn_rankings = [{'disease': disease, 'count': count} for disease, count in sorted_corn_diseases]
    rice_rankings = [{'disease': disease, 'count': count} for disease, count in sorted_rice_diseases]
    beans_rankings = [{'disease': disease, 'count': count} for disease, count in sorted_beans_diseases]

    # Return rankings for each crop
    return jsonify({
        'Tomato': tomato_rankings,
        'Corn': corn_rankings,
        'Rice': rice_rankings,
        'Beans': beans_rankings
    })

@app.route('/plant_disease_rankings_city', methods=['POST'])
def get_plant_disease_rankings_by_city():
    # Get latitude and longitude from the request
    latitude = request.form.get('latitude')
    longitude = request.form.get('longitude')

    # Check if latitude and longitude are provided
    if not latitude or not longitude:
        return jsonify({"error": "Latitude and longitude are required."}), 400

    # Convert latitude and longitude to floats
    latitude_float = float(latitude)
    longitude_float = float(longitude)

    # Find the city using reverse geocoding
    city = reverse_geocode_city(latitude_float, longitude_float)

    # Get plant disease counts for each crop in the city
    tomato_records = TomatoData.query.filter_by(city=city).all()
    corn_records = CornData.query.filter_by(city=city).all()
    rice_records = RiceData.query.filter_by(city=city).all()
    beans_records = BeansData.query.filter_by(city=city).all()

    # Initialize defaultdicts to store plant disease counts for each crop
    tomato_counts = defaultdict(int)
    corn_counts = defaultdict(int)
    rice_counts = defaultdict(int)
    beans_counts = defaultdict(int)

    # Function to update counts for each crop
    def update_counts(records, counts):
        for record in records:
            counts[record.plant_disease] += 1

    # Update counts for each crop
    update_counts(tomato_records, tomato_counts)
    update_counts(corn_records, corn_counts)
    update_counts(rice_records, rice_counts)
    update_counts(beans_records, beans_counts)

    # Sort the diseases by count in descending order for each crop
    sorted_tomato_diseases = sorted(tomato_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_corn_diseases = sorted(corn_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_rice_diseases = sorted(rice_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_beans_diseases = sorted(beans_counts.items(), key=lambda x: x[1], reverse=True)

    # Create a list of dictionaries containing disease rankings for each crop
    tomato_rankings = [{'disease': disease, 'count': count} for disease, count in sorted_tomato_diseases]
    corn_rankings = [{'disease': disease, 'count': count} for disease, count in sorted_corn_diseases]
    rice_rankings = [{'disease': disease, 'count': count} for disease, count in sorted_rice_diseases]
    beans_rankings = [{'disease': disease, 'count': count} for disease, count in sorted_beans_diseases]

    # Return rankings for each crop
    return jsonify({
        'Tomato': tomato_rankings,
        'Corn': corn_rankings,
        'Rice': rice_rankings,
        'Beans': beans_rankings
    })


if __name__ == '__main__':
    app.run()
