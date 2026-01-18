FROM tensorflow/tensorflow:latest-gpu

ARG TIMEZONE="Europe/Madrid"

# Set timezone
RUN ln -snf /usr/share/zoneinfo/${TIMEZONE} /etc/localtime && echo ${TIMEZONE} > /etc/timezone

ARG USER_ID=1000
ARG GROUP_ID=1000

# Arguments for the system user and group
ARG SYSTEM_USER="mario"
ARG SYSTEM_GROUP="mario"

# Asigna grupo y usuario en contenedor para no tener que estar cambiando propietario a los archivos creados desde el contenedor
RUN addgroup --gid ${GROUP_ID} ${SYSTEM_GROUP}
RUN adduser --disabled-password --gecos '' --uid ${USER_ID} --gid ${GROUP_ID} ${SYSTEM_USER}


WORKDIR /app
RUN apt-get update && apt-get install -y libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

# Copiamos el modelo y el script
COPY model.keras .
COPY app.py .

# Creamos la carpeta de uploads dentro del contenedor
RUN mkdir uploads

EXPOSE 5000

USER ${SYSTEM_USER}

CMD ["python", "app.py"]