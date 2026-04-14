import os
import gdown
from flask import Flask, render_template, request
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd

# -------------------- MODEL DOWNLOAD --------------------
MODEL_PATH = "plant_disease_model_1_latest.pt"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    url = "https://drive.google.com/uc?id=1r4rYEvRE5lOyvg0ia8b2QrJQVKeofVgr"
    gdown.download(url, MODEL_PATH, quiet=False)

# -------------------- LOAD DATA --------------------
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

# -------------------- LOAD MODEL --------------------
model = CNN.CNN(39)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

# -------------------- PREDICTION FUNCTION --------------------
def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))

    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)

    return index

# -------------------- FLASK APP --------------------
app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename

        upload_folder = "static/uploads"
        os.makedirs(upload_folder, exist_ok=True)

        file_path = os.path.join(upload_folder, filename)
        image.save(file_path)

        pred = prediction(file_path)

        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]

        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]

        return render_template(
            'submit.html',
            title=title,
            desc=description,
            prevent=prevent,
            image_url=image_url,
            pred=pred,
            sname=supplement_name,
            simage=supplement_image_url,
            buy_link=supplement_buy_link
        )

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template(
        'market.html',
        supplement_image=list(supplement_info['supplement image']),
        supplement_name=list(supplement_info['supplement name']),
        disease=list(disease_info['disease_name']),
        buy=list(supplement_info['buy link'])
    )

# -------------------- RUN --------------------
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)