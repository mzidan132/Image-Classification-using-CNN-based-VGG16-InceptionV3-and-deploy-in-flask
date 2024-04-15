from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import uuid
import threading
import time
app = Flask(__name__)
model = keras.models.load_model('VGG16.h5')
models = keras.models.load_model('VGG16.h5')
class_names = ['Black Spot', 'Cercospora Leaf Spot', 'Downy Mildew', 'Fresh Leaf', 'Powdery Mildew', 'Rose Botrytis Blight', 'Rose Slug']
class_name = ['Black Spot', 'Cercospora Leaf Spot', 'Downy Mildew', 'Fresh Leaf', 'Powdery Mildew', 'Rose Botrytis Blight', 'Rose Slug']
# Function to generate a unique filename using UUID
def generate_unique_filename(filename):
    _, extension = os.path.splitext(filename)
    unique_filename = str(uuid.uuid4()) + extension
    return unique_filename

@app.route('/')
def home():
    image_exists = os.path.exists('static/temp.JPG')
    if image_exists:
        image_url = f'/static/temp.JPG'
    else:
        image_url = None

    return render_template('index.html', image_exists=image_exists, image_url=image_url, background_image_url="/static/pf.jpg")

@app.route('/predictions', methods=["POST"])
def predictions():
    img = request.files['img']
    
    # Generate a unique filename
    unique_filename = generate_unique_filename(img.filename)
    image_url = os.path.join('static', 'images', unique_filename)
   
    img.save(image_url)
  
    # Load and preprocess the image for prediction
    img = image.load_img(image_url, target_size=(224, 224))
    x = image.img_to_array(img) / 255
    resized_img_np = np.expand_dims(x, axis=0)
    
    # Make predictions using the model
    prediction = models.predict(resized_img_np)
    pred_class_index = np.argmax(prediction)
    pred_class_name = class_names[pred_class_index]

    # Render the template with the prediction results
    return render_template("index.html", datas=prediction, class_name=pred_class_name, image_url=image_url, background_image_url="/static/pf.jpg")

@app.route('/prediction', methods=["POST"])
def prediction():
    img = request.files['img']
    
    # Generate a unique filename
    unique_filename = generate_unique_filename(img.filename)
    image_urls = os.path.join('static', 'images', unique_filename)
    
    # Save the image
    img.save(image_urls)
  
    # Load and preprocess the image for prediction
    img = image.load_img(image_urls, target_size=(224, 224)) #299,299 for inceptionV3 model
    x = image.img_to_array(img) / 255
    resized_img_np = np.expand_dims(x, axis=0)
    
    # Make predictions using the model
    prediction = model.predict(resized_img_np)
    pred_class_index = np.argmax(prediction)
    pred_class_name = class_name[pred_class_index]

    # Render the template with the prediction results
    return render_template("index.html", data=prediction, class_name=pred_class_name, image_url=image_urls, background_image_url="/static/pf.jpg")

def delete_images_after_delay():
    while True:
        time.sleep(40)  # Wait 1 day
        image_folder = 'static/images'
        for filename in os.listdir(image_folder):
            file_path = os.path.join(image_folder, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")

# Flask route to delete images after 2 minutes
@app.route('/delete', methods=['GET'])
def delete():
    threading.Thread(target=delete_images_after_delay).start()
    return jsonify({"message": "Images will be deleted continuously after 2 minutes."})

if __name__ == '__main__':
    app.run(port=5000)

