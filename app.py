from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
from PIL import Image
import io

app = Flask(__name__)

# Load models
ml_model = pickle.load(open('ml_model.pkl', 'rb'))
dl_model = tf.keras.models.load_model('best_model.keras')

# Helper to preprocess image for DenseNet (assuming 224x224x3 input)
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    if len(img_array.shape) == 2:  # grayscale
        img_array = np.stack([img_array]*3, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image

@app.route('/', methods=['GET'])
def index():
    symptoms = [
        "Symptom_Body ache", "Symptom_Cough", "Symptom_Fatigue", "Symptom_Fever",
        "Symptom_Headache", "Symptom_Runny nose", "Symptom_Shortness of breath", "Symptom_Sore throat"
    ]
    return render_template("index.html", symptoms=symptoms)

@app.route('/predict_symptoms', methods=['POST'])
def predict_symptoms():
    try:
        features = [
            int(request.form.get("Age")),
            int(request.form.get("Gender")),
            int(request.form.get("Heart_Rate_bpm")),
            float(request.form.get("Body_Temperature_C")),
            int(request.form.get("Oxygen_Saturation_%")),
            int(request.form.get("Symptom_Body ache")),
            int(request.form.get("Symptom_Cough")),
            int(request.form.get("Symptom_Fatigue")),
            int(request.form.get("Symptom_Fever")),
            int(request.form.get("Symptom_Headache")),
            int(request.form.get("Symptom_Runny nose")),
            int(request.form.get("Symptom_Shortness of breath")),
            int(request.form.get("Symptom_Sore throat")),
        ]

        feature_names = [
            "Age", "Gender", "Heart_Rate_bpm", "Body_Temperature_C", "Oxygen_Saturation_%",
            "Symptom_Body ache", "Symptom_Cough", "Symptom_Fatigue", "Symptom_Fever",
            "Symptom_Headache", "Symptom_Runny nose", "Symptom_Shortness of breath",
            "Symptom_Sore throat"
        ]

        ml_input_df = pd.DataFrame([features], columns=feature_names)
        ml_pred = ml_model.predict(ml_input_df)[0]

        ml_mapping = {
            0: "Hospitalization and medication",
            1: "Rest and medication",
            2: "Rest and fluids"
        }
        ml_result = ml_mapping.get(ml_pred, "Unknown ML Prediction")

        # Add hardcoded relief suggestions for symptoms
        relief_mapping = {
            "Symptom_Body ache": "Apply a warm compress and rest well.",
            "Symptom_Cough": "Use honey in warm water or a cough suppressant.",
            "Symptom_Fatigue": "Get adequate sleep and stay hydrated.",
            "Symptom_Fever": "Take paracetamol and stay cool.",
            "Symptom_Headache": "Use a cold pack and stay in a dark room.",
            "Symptom_Runny nose": "Use a saline nasal spray.",
            "Symptom_Shortness of breath": "Sit upright and practice slow deep breathing.",
            "Symptom_Sore throat": "Gargle with warm salt water.",
        }

        selected_symptoms = dict(zip(feature_names[5:], features[5:]))  # just symptom parts
        reliefs = [
            relief_mapping[key]
            for key, val in selected_symptoms.items()
            if val == 1 and key in relief_mapping
        ]

        return jsonify({
            'result': f"Health Advice: {ml_result}",
            'relief': reliefs
        })

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict_xray', methods=['POST'])
def predict_xray():
    try:
        file = request.files['image']
        img = Image.open(file.stream)
        dl_input = preprocess_image(img)
        dl_pred = dl_model.predict(dl_input)
        dl_label = np.argmax(dl_pred)

        dl_mapping = {
            0: "Bacterial Pneumonia",
            1: "Normal",
            2: "Viral Pneumonia"
        }
        dl_result = dl_mapping.get(dl_label, "Unknown DL Prediction")
        return jsonify({'result': f"X-ray Diagnosis: {dl_result}"})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)