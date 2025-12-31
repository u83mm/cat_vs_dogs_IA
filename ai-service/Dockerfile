FROM tensorflow/tensorflow:latest-gpu

WORKDIR /app
RUN apt-get update && apt-get install -y libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

# Copiamos el modelo y el script
COPY perros_gatos_master.keras .
COPY app.py .

# Creamos la carpeta de uploads dentro del contenedor
RUN mkdir uploads

EXPOSE 5000

CMD ["python", "app.py"]