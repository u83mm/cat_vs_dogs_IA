## Creación del modelo
### Obtención y preparación del Dataset
<p>Una vez que tenemos Jupyterlab en marcha, creamos la carpeta <code>cats_vs_dogs</code> y dentro de la misma pondremos las carpetas <code>test</code> y <code>train</code> que vienen en el Dataset que obtenemos, en nuestro caso lo hacemos del siguiente link:</p>

```
https://www.kaggle.com/datasets/moazeldsokyx/dogs-vs-cats
```
### Jupyterlab

#### Preparación del entorno

```
import os
from tensorflow.keras import layers, models

# set the environment
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

# supress low level warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

# set only GPU VRAM needed
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# set the images
img_size = (160, 160)
batch_size = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    'cats_vs_dogs/train',
    image_size=img_size,
    batch_size=batch_size,
    label_mode='binary'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    'cats_vs_dogs/validation',
    image_size=img_size,
    batch_size=batch_size,
    label_mode='binary'
)

# Data augmentation
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# define the CNN
model = models.Sequential([
    layers.Input(shape=(160, 160, 3)),

    # apply data augmentation
    data_augmentation,
        
    # scale pixels
    layers.Rescaling(1./255),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.7),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```
#### Entrenamiento del modelo

```
# train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=12,
    #callbacks=[early_stopping] 
)
```
#### Comprobación de la efectividad del modelo

```
import matplotlib.pyplot as plt

# show graphics to look at the model precission
plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model precission')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.show()
```
#### Guardamos el modelo

```
# save the model
model.save('model.keras')
print('¡Modelo de visión artificial guardado!')
```
#### Definición de test
<p>Definimos un test para poder probar nuestro modelo sin tener que volver a entrenar el modelo.</p>

```
# Load the model and test
import os
import numpy as np
from tensorflow.keras.preprocessing import image

# set the environment
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

# supress low level warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

# load the model
model = tf.keras.models.load_model('dogs_vs_cats_model1.keras')

def predict_animal(img_path):
    # load the image
    img = image.load_img(img_path, target_size=(160, 160))
    
    # convert to an array and add 'batch' dimension (Tensorflow wait for [batch, high, width, channels]
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    #img_array = img_array / 255.0
    
    # predict (near 1 = DOG, near 0 = CAT)
    prediction = model.predict(img_array)
    
    if prediction[0] > 0.5:
        score = prediction[0][0]
        print(f"It's a DOG (Trust: {score:.2%})")
    else:
        score = 1 - prediction[0][0]
        print(f"It's a CAT (Trust: {score:.2%})")
```
#### Obtenemos el resultado
<p>Previamente descargamos algunas imágenes de internet para probar el modelo.</p>

```
# test the model
img_path = "perro3.jpg"
predict_animal(img_path)
```
<hr>

#### Explicación del código

```
# set the images
img_size = (160, 160)
batch_size = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    'cats_vs_dogs/train',
    image_size=img_size,
    batch_size=batch_size,
    label_mode='binary'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    'cats_vs_dogs/test',
    image_size=img_size,
    batch_size=batch_size,
    label_mode='binary'
)
```
<p>Esta parte del código se encarga de <strong>preparar y cargar las imágenes</strong> desde tu computadora hacia el programa para que el modelo las pueda procesar.</p>
<p>Aquí tienes el desglose paso a paso:</p>
<ol>
    <li>
    Configuración de parámetros
    <p>Antes de cargar las fotos, se definen dos variables clave:</p>
    <p><strong>img_size = (160, 160):</strong> Todas las imágenes originales pueden tener tamaños distintos. Esta línea asegura que, al leerlas, todas se reescalen a <strong>160x160 píxeles</strong> para que el modelo reciba datos uniformes.</p>
    <p><strong>batch_size = 32:</strong> El modelo no procesa todas las fotos a la vez (consumiría demasiada memoria). En su lugar, las toma en grupos o "lotes" de <strong>32 imágenes</strong>.</p>
    </li>    
    <li>
    Carga del Dataset de Entrenamiento (train_ds)
    <p>La función image_dataset_from_directory automatiza la lectura de carpetas:</p>
    <p><strong>'cats_vs_dogs/train':</strong> Es la ruta de la carpeta donde están las fotos de entrenamiento. Keras asume que dentro de esta carpeta hay subcarpetas (ej. /cats y /dogs) y usa esos nombres como etiquetas.</p>
    <p><strong>image_size=img_size:</strong> Aplica el tamaño de 160x160 definido antes.</p>
    <p><strong>batch_size=batch_size:</strong> Agrupa las imágenes en los lotes de 32 mencionados.</p>
    <p><strong>label_mode='binary':</strong> Indica que solo hay dos categorías (gato o perro). Esto hará que las etiquetas sean simplemente 0 o 1, ideal para una clasificación de "esto o aquello".</p>
    </li>
    <li>
    Carga del Dataset de Validación (val_ds)
    <p>Se repite el mismo proceso pero apuntando a la carpeta <strong>'cats_vs_dogs/validation'</strong>. Estos datos se mantienen separados y sirven para que, durante el entrenamiento, el modelo pueda ser evaluado con imágenes que "nunca ha visto", permitiéndote saber si realmente está aprendiendo o solo memorizando.</p>
    <p><strong>En resumen:</strong> Estas líneas convierten tus carpetas de fotos en objetos de datos listos para ser "leídos" eficientemente por la tarjeta gráfica (GPU) durante el entrenamiento.</p>
    </li>
</ol>

```
# Data augmentation
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])
```
<p>Esta sección del código define una técnica llamada <strong>Aumento de Datos</strong> (<em>o Data Augmentation</em>). Su objetivo principal es crear "variedad artificial" en tus imágenes para que el modelo sea más robusto y no se memorice las fotos exactas del entrenamiento.</p>
<p>Aquí tienes el desglose de cada parte:</p>
<ol>
    <li>
    models.Sequential([...])
    <p>Crea una pequeña secuencia de pasos que se aplicarán a cada imagen justo antes de entrar a la red neuronal. Es como una "aduana" por la que pasan las fotos para ser transformadas.</p>
    </li>
    <li>
    layers.RandomFlip("horizontal")
    <p><strong>Qué hace:</strong> Gira la imagen de izquierda a derecha de forma aleatoria (como un espejo).</p>
    <p><strong>Por qué sirve:</strong> Un perro sigue siendo un perro aunque esté mirando hacia el otro lado. Esto ayuda al modelo a no depender de la dirección hacia la que mira el animal.</p>
    </li>
    <li>
    layers.RandomRotation(0.1)
    <p><strong>Qué hace:</strong> Rota la imagen aleatoriamente hasta un <strong>10%</strong> (aproximadamente 36 grados) en cualquier dirección.</p>
    <p><strong>Por qué sirve:</strong> En la vida real, las fotos no siempre están perfectamente niveladas. Esto enseña al modelo a reconocer al animal aunque la cámara esté un poco inclinada.</p>
    </li>
    <li>
    layers.RandomZoom(0.1)
    <p><strong>Qué hace:</strong> Aplica un zoom (hacia adentro o hacia afuera) de forma aleatoria de hasta un <strong>10%</strong>.</p>
    <p><strong>Por qué sirve:</strong> Ayuda al modelo a reconocer objetos aunque aparezcan más cerca o más lejos de la cámara.</p>
    </li>
</ol>
<hr>
<p>¿Para qué sirve todo esto en conjunto?</p>
<p>El <strong>Aumento de Datos</strong> es la mejor herramienta contra el <strong>Overfitting</strong> (sobreajuste). Sin esto, si todas tus fotos de perros tuvieran al perro a la derecha, el modelo podría aprender erróneamente que "perro = algo a la derecha". Al girar, rotar y hacer zoom, obligas al modelo a aprender <strong>las características reales</strong> (orejas, hocico, ojos) en lugar de solo la posición de los píxeles.</p>
<p><strong>Nota técnica:</strong> Estas transformaciones solo ocurren durante el entrenamiento y son aleatorias en cada época, por lo que el modelo casi nunca ve la misma imagen exacta dos veces.</p>

```
# define the CNN
model = models.Sequential([
    layers.Input(shape=(160, 160, 3)),

    # apply data augmentation
    data_augmentation,
        
    # scale pixels
    layers.Rescaling(1./255),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.7),
    layers.Dense(1, activation='sigmoid')
])
```
<p>Esta es la arquitectura de tu <strong>Red Neuronal Convencional (CNN)</strong>. Aquí es donde se define la "inteligencia" que aprenderá a distinguir entre perros y gatos. Se lee como una línea de ensamblaje, donde la imagen entra por arriba y el resultado sale por abajo:</p>

<ol>
    <li>
    Entrada y Preprocesamiento
    <p><strong>layers.Input(shape=(160, 160, 3)):</strong> Define el tamaño de entrada. 160x160 son los píxeles y 3 indica los canales de color (Rojo, Verde, Azul - RGB).</p>
    <p><strong>data_augmentation:</strong> Aquí insertas el bloque que explicamos antes. Cada imagen que entre será rotada o girada aleatoriamente antes de ser analizada.</p>
    <p><strong>layers.Rescaling(1./255):</strong> Las imágenes digitales tienen píxeles de 0 a 255. Las redes neuronales trabajan mejor con números pequeños; esta línea divide todo por 255 para que los valores queden entre <strong>0 y 1</strong>.</p>
    </li>
    <li>
    Extracción de Características (Capas Convencionales)
    <p>Esta es la parte "visual" del modelo:</p>
    <p><strong>layers.Conv2D(32, (3, 3), activation='relu'):</strong> Es un "filtro" que recorre la imagen buscando patrones simples (bordes, líneas). Crea 32 mapas de características distintos.</p>
    <p><strong>layers.MaxPooling2D(2, 2):</strong> Reduce el tamaño de la imagen a la mitad. Se queda solo con la información más importante para que el modelo sea más rápido y no se pierda en detalles irrelevantes.</p>
    <p><strong>layers.Conv2D(64, ...):</strong> Otro filtro, pero ahora busca patrones más complejos (curvas, formas de ojos o de orejas). Al tener 64 filtros, puede "ver" más detalles.</p>
    </li>
    <li>
    Clasificación (Capas Densas)
    <p>Aquí es donde el modelo toma la información visual y toma una decisión:</p>
    <p><strong>layers.Flatten():</strong> Convierte los mapas de características bidimensionales en una sola lista larga de números (un vector plano).</p>
    <p><strong>layers.Dense(128, activation='relu'):</strong> Una capa de 128 "neuronas" que conectan todos los patrones detectados para intentar entender qué animal es.</p>
    <p><strong>layers.Dropout(0.5):</strong> Es una técnica de seguridad. "Apaga" el 50% de las neuronas aleatoriamente en cada paso del entrenamiento. Esto obliga al modelo a no depender de una sola neurona y a ser más robusto (evita el overfitting).</p>
    <p><strong>layers.Dense(1, activation='sigmoid'):</strong> La neurona final.</p>
    <p>Usa <strong>sigmoid</strong> porque solo hay dos opciones. Devolverá un número entre <strong>0 y 1</strong>. Si el resultado es cercano a 0, el modelo cree que es una clase (ej. gato); si es cercano a 1, cree que es la otra (ej. perro).</p>    
    </li>
</ol>

<p>Resumen del flujo:</p>
<p>1. <strong>Entra la imagen</strong> → 2. <strong>Se transforma</strong> (aumento) → 3. <strong>Se normaliza</strong> (0-1) → 4. <strong>Se detectan formas</strong> (Conv2D) → 5. <strong>Se simplifica</strong> (Pooling) → 6. <strong>Se analiza la información</strong> (Dense) → 7. <strong>Se da un resultado final</strong>.</p>

```
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```
<p>Esta parte del código es la <strong>fase de configuración</strong> del modelo. Antes de empezar a entrenar con los datos, debes decirle al modelo "cómo" debe aprender y "cómo" vas a medir si lo está haciendo bien.</p>
<p>Aquí tienes el desglose de los tres pilares del aprendizaje:</p>

<ol>
    <li>
        optimizer='adam' (El Cerebro)
        <p>El optimizador es el algoritmo que se encarga de actualizar los pesos de la red neuronal para reducir el error.</p>
        <p><strong>Adam</strong> es el estándar de la industria hoy en día (2025). Es muy eficiente porque ajusta automáticamente la "velocidad" de aprendizaje (learning rate). Si el modelo está lejos de la respuesta, da pasos grandes; si está cerca, da pasos pequeños para no pasarse de largo.</p>
    </li>
    <li>
        loss='binary_crossentropy' (La Regla de Medir)
        <p>La "función de pérdida" (loss) es la forma en que el modelo calcula <strong>qué tan equivocado está</strong>.</p>
        <p><strong>binary_crossentropy</strong> se usa específicamente cuando tienes una <strong>clasificación binaria</strong> (solo dos opciones, como gato o perro).</p>
        <p>Si el modelo está muy seguro de que una foto es un gato y resulta ser un perro, esta función le dará una "penalización" muy alta para que aprenda del error drásticamente.</p>
    </li>
    <li>
        metrics=['accuracy'] (El Reporte)
        <p>Aquí defines qué estadísticas quieres ver mientras el modelo entrena.</p>
        <p><strong>accuracy (Precisión):</strong> Es el porcentaje de aciertos. Por ejemplo, si de 100 imágenes el modelo adivina 90 correctamente, verás un 0.90 en tu pantalla durante el entrenamiento. Es la métrica más fácil de entender para nosotros los humanos para saber si el modelo está funcionando.</p>
    </li>
</ol>

<p>En resumen, con esta línea le estás diciendo al programa:</p>
<p><em>"Usa el algoritmo Adam para mejorar, mide tus errores con Binary Crossentropy y
muéstrame en pantalla el porcentaje de aciertos después de cada paso."</em></p>

<hr>

```
# train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=12,
    #callbacks=[early_stopping] 
)
```
<p>Este fragmento de código utiliza la biblioteca TensorFlow/Keras para entrenar una red neuronal.</p>

<p>Aquí tienes el desglose paso a paso:</p>

<ol>
    <li><p><strong>history =:</strong>El método fit devuelve un objeto llamado History. Este objeto registra las métricas (como la precisión y la pérdida) obtenidas al final de cada época, lo que te permite graficar el rendimiento del modelo después. Puedes consultar la documentación oficial de Keras para ver qué datos almacena.</p></li>
    <li><p><strong>model.fit(...):</strong> Es la función principal para iniciar el entrenamiento. "Ajusta" (fit) los parámetros del modelo (pesos y sesgos) para que aprenda a predecir correctamente basándose en los datos proporcionados.</p></li>
    <li><p><strong>train_ds:</strong> Representa el <strong>conjunto de datos de entrenamiento</strong>. Es la fuente de información de la cual el modelo aprenderá los patrones. Generalmente es un objeto de tipo tf.data.Dataset.</p></li>
    <li><p><strong>validation_data=val_ds:</strong> Define el <strong>conjunto de datos de validación</strong>. Al final de cada época, el modelo se prueba con estos datos (que no ha visto durante el entrenamiento) para verificar si está generalizando bien o si está sufriendo de overfitting (sobreajuste).</p></li>
    <li><p><strong>epochs=10:</strong> Indica el <strong>número de épocas</strong>. Una época es una pasada completa de todos los datos de entrenamiento a través de la red neuronal. En este caso, el proceso se repetirá 10 veces.</p></li>
</ol>

```
import matplotlib.pyplot as plt

# show graphics to look at the model precission
plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model precission')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.show()
```
<p>Esta parte del código se encarga de la <strong>visualización de resultados</strong>. Utiliza la biblioteca Matplotlib para crear una gráfica que te permite evaluar visualmente si tu modelo aprendió correctamente o si tiene problemas.</p>
<p>Aquí tienes el desglose paso a paso:</p>
<ol>
    <li>
        import matplotlib.pyplot as plt
        <p>Importa la herramienta de dibujo estándar de Python. Se le asigna el alias plt para que sea más rápido de escribir.</p>
    </li>
    <li>
        plt.plot(history.history['accuracy'], ...)
        <p><strong>¿Qué hace?:</strong> Dibuja la línea de <strong>precisión de entrenamiento</strong>.</p>
        <p><strong>De dónde viene:</strong> Recuerdas que al principio guardamos el entrenamiento en una variable llamada history. Esta variable contiene un diccionario con el registro de qué tan bien le fue al modelo en cada una de las 10 épocas.</p>
    </li>
    <li>
        plt.plot(history.history['val_accuracy'], ...)
        <p><strong>¿Qué hace?:</strong> Dibuja la línea de <strong>precisión de validación</strong>.</p>
        <p><strong>Por qué es importante:</strong> Esta es la línea clave. Muestra qué tan bien se comporta el modelo con imágenes que <strong>no utilizó para entrenar</strong>.</p>
    </li>
    <li>
        Configuración de la gráfica (Títulos y Etiquetas)
        <p><strong>plt.title, plt.ylabel, plt.xlabel:</strong> Añaden el título principal ("Model precision") y los nombres a los ejes (el eje vertical es el porcentaje de acierto y el horizontal son las épocas o iteraciones).</p>
        <p><strong>plt.legend():</strong> Muestra el recuadro que indica qué color de línea corresponde a "Training" y cuál a "Validation".</p>
    </li>
    <li>
        plt.show()
        <p>Es la orden final que abre la ventana y proyecta la gráfica en tu pantalla.</p>
    </li>
</ol>
<hr>
<p>¿Cómo interpretar esta gráfica en 2025?</p>
<p>Al ejecutar esto, verás dos líneas. Lo ideal es que ambas suban juntas. Aquí hay dos escenarios comunes:</p>

<ol>
    <li>
        <p><strong>Escenario Ideal:</strong> Ambas líneas suben y terminan cerca una de la otra (ej. 90%). Esto significa que tu modelo aprendió a identificar perros y gatos en general.</p>
    </li>
    <li>
        <p><strong>Overfitting (Sobreajuste):</strong> La línea de Training sube al 99% pero la de Validation se queda estancada abajo o empieza a bajar. Esto significa que el modelo <strong>memorizó</strong> las fotos de entrenamiento pero no sabe reconocer fotos nuevas.</p>
    </li>
</ol>

```
# Load the model and test
import os
import numpy as np
from tensorflow.keras.preprocessing import image

# set the environment
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

# supress low level warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

# load the model
model = tf.keras.models.load_model('model.keras')

def predict_animal(img_path):
    # load the image
    img = image.load_img(img_path, target_size=(160, 160))
    
    # convert to an array and add 'batch' dimension (Tensorflow wait for [batch, high, width, channels]
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    #img_array = img_array / 255.0
    
    # predict (near 1 = DOG, near 0 = CAT)
    prediction = model.predict(img_array)
    
    if prediction[0] > 0.5:
        score = prediction[0][0]
        print(f"It's a DOG (Trust: {score:.2%})")
    else:
        score = 1 - prediction[0][0]
        print(f"It's a CAT (Trust: {score:.2%})")
```
<p>Esta sección del código es la <strong>Inferencia</strong>. Sirve para usar el conocimiento que el modelo adquirió durante el entrenamiento y aplicarlo a imágines nuevas.</p>
<p>Aquí tienes el desglose paso a paso:</p>

<ol>
    <li>
        Configuración del Entorno y Carga
        <p><strong>os.environ[...]:</strong> Estas líneas configuran cómo TensorFlow interactúa con tu hardware. El nivel de log '2' se usa para limpiar la consola, ocultando mensajes informativos y dejando solo los errores graves[1].</p>
        <p><strong>tf.keras.models.load_model('...'):</strong> Esta es la función clave. Carga el archivo .keras que guardaste previamente (que contiene la estructura y los "pesos" o memoria del modelo). Ya no necesitas volver a entrenar; el modelo ya sabe qué buscar [2].</p>
    </li>
    <li>
        Preparación de la imagen (predict_animal)
        <p>Antes de que el modelo pueda "mirar" una foto, esta debe pasar por un proceso de transformación:</p>
        <p><strong>image.load_img(..., target_size=(160, 160)):</strong> Abre la foto y la fuerza a tener el tamaño de 160x160 píxeles, exactamente igual a como entrenaste al modelo.</p>
        <p><strong>img_to_array:</strong> Convierte los colores de la imagen en una matriz de números que la computadora entiende.</p>
        <p><strong>np.expand_dims(..., axis=0):</strong> TensorFlow no acepta una sola imagen suelta; espera un "lote" (batch). Esta línea convierte la imagen de una sola pieza a una "lista de una imagen", dándole la forma (1, 160, 160, 3).</p>
    </li>
    <li>
        La Predicción y la Lógica de Decisión
        <p><strong>model.predict(img_array):</strong> El modelo analiza los píxeles y devuelve un número decimal entro 0 y 1.</p>
        <p></p>
        <p><strong>Interpretación del resultado:</strong> Como usamos una activación sigmoid al final del modelo, el resultado es una probabilidad.</p>
        <p><strong>if prediction[0] > 0.5:</strong> Si el número es mayor a 0.5, el modelo está más seguro de que es un <strong>perro</strong>. Cuanto más cerca de 1, mayor es la confianza.</p>
        <p><strong>else:</strong> Si es menor a 0.5, el modelo determina que es un <strong>gato</strong>.</p>
    </li>
    <li>
        Cálculo de Confianza (score)
        <p>Para el perro, el valor directo es la confianza.</p>
        <p>Para el gato, se resta de 1 (ej: si sale 0.1, el modelo tiene un 1 - 0.1 = 0.9 o 90% de confianza de que es un gato).</p>
        <p><strong>{score:.2%}:</strong> Es un formato de texto que convierte el número (ej: 0.854) en un porcentaje legible (85.40%).</p>
    </li>
</ol>