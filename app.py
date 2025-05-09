from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Load the trained model
classifier = load_model('thermal_new.h5')

# Define class names manually if dataset.class_names is not available
class_names = ['Fresh','Rotten']  # Replace with actual class names

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    predicted_class = None
    file_path = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            
            # Process image
            test_img = image.load_img(file_path, target_size=(150, 150))
            test_img = image.img_to_array(test_img)
            test_img = np.expand_dims(test_img, axis=0)
            
            # Predict
            result = classifier.predict(test_img)
            predicted_class = class_names[np.argmax(result)]
            
    return render_template('index.html', file_path=file_path, predicted_class=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)