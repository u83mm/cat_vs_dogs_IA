import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.preprocessing import image

# setup for not use all the GPU memory
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            # Esto evita que TF reserve toda la VRAM de golpe
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Configuración de memoria dinámica activada")
    except RuntimeError as e:
        print(e)

app = Flask(__name__)
CORS(app)

# Ruta dentro del contenedor de Python donde se monta el volumen compartido
UPLOAD_FOLDER = '/app/uploads'

# Cargamos el modelo MAESTRO
MODEL_PATH = 'perros_gatos_master.keras'
model = tf.keras.models.load_model(MODEL_PATH)

def prepare_image(img_path):
    # Usamos 160x160 como acordamos para evitar Warnings
    img = image.load_img(img_path, target_size=(160, 160))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # MobileNetV2.preprocess_input ya está dentro del modelo (capa Lambda), 
    # así que aquí solo pasamos el array.
    return img_array

@app.route('/predict', methods=['POST'])

def predict():
    data = request.get_json()
    
    if not data or 'filename' not in data:
        return jsonify({'error': 'No se proporcionó el nombre del archivo'}), 400
    
    # Construimos la ruta completa al archivo en el volumen compartido
    img_path = os.path.join(UPLOAD_FOLDER, data['filename'])
    
    if not os.path.exists(img_path):
        return jsonify({'error': f'El archivo {data["filename"]} no existe en el volumen'}), 404

    try:
        # Procesar y predecir
        img_ready = prepare_image(img_path)
        prediction = model.predict(img_ready)
        
        # Interpretación
        prob = float(prediction[0][0])
        label = "Perro" if prob > 0.5 else "Gato"
        confidence = prob if prob > 0.5 else 1 - prob
        
        return jsonify({
            'class': label,
            'confidence': f"{confidence:.2%}",
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Escuchamos en todas las interfaces para que Docker lo vea
    app.run(host='0.0.0.0', port=5000)