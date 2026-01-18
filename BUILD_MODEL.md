## Model Creation
### Dataset Acquisition and Preparation
<p>Once we have Jupyterlab up and running, we create the folder <code>cats_vs_dogs</code> and inside it, we will place the folders <code>test</code> and <code>train</code> that come with the Dataset we obtain, in our case, we do it from the following link:</p>

```
https://www.kaggle.com/datasets/moazeldsokyx/dogs-vs-cats
```
### Jupyterlab

#### Environment Preparation

```
import os
from tensorflow.keras import layers, models

# set the environment
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

# suppress low level warnings
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
    'cats_vs_dogs/test',
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
#### Model Training

```
# train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=12,
    #callbacks=[early_stopping] 
)
```
#### Checking Model Effectiveness

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
#### Saving the Model

```
# save the model
model.save('model.keras')
print('¡Modelo de visión artificial guardado!')
```
#### Test Definition
<p>We define a test to be able to try our model without having to retrain the model.</p>

```
# Load the model and test
import os
import numpy as np
from tensorflow.keras.preprocessing import image

# set the environment
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

# suppress low level warnings
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
#### Obtaining the Result
<p>Previously, we downloaded some images from the internet to test the model.</p>

```
# test the model
img_path = "perro3.jpg"
predict_animal(img_path)
```
<hr>

#### Code Explanation

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
<p>This part of the code is responsible for <strong>preparing and loading the images</strong> from your computer to the program so that the model can process them.</p>
<p>Here is the step-by-step breakdown:</p>
<ol>
    <li>
    Parameter Setting
    <p>Before loading the photos, two key variables are defined:</p>
    <p><strong>img_size = (160, 160):</strong> All original images may have different sizes. This line ensures that, when reading them, they are all re-scaled to <strong>160x160 pixels</strong> so that the model receives uniform data.</p>
    <p><strong>batch_size = 32:</strong> The model does not process all the photos at once (it would consume too much memory). Instead, it takes them in groups or "batches" of <strong>32 images</strong>.</p>
    </li>    
    <li>
    Loading the Training Dataset (train_ds)
    <p>The image_dataset_from_directory function automates the reading of folders:</p>
    <p><strong>'cats_vs_dogs/train':</strong> This is the path to the folder where the training photos are located. Keras assumes that within this folder there are subfolders (e.g., /cats and /dogs) and uses those names as labels.</p>
    <p><strong>image_size=img_size:</strong> Applies the size of 160x160 defined earlier.</p>
    <p><strong>batch_size=batch_size:</strong> Groups the images into the mentioned batches of 32.</p>
    <p><strong>label_mode='binary':</strong> Indicates that there are only two categories (cat or dog). This will make the labels simply 0 or 1, ideal for a "this or that" classification.</p>
    </li>
    <li>
    Loading the Validation Dataset (val_ds)
    <p>The same process is repeated but pointing to the folder <strong>'cats_vs_dogs/validation'</strong>. This data is kept separate and serves so that, during training, the model can be evaluated with images that it has "never seen before", allowing you to know if it is really learning or just memorizing.</p>
    <p><strong>In summary:</strong> These lines convert your photo folders into data objects ready to be "read" efficiently by the graphics card (GPU) during training.</p>
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
<p>This section of the code defines a technique called <strong>Data Augmentation</strong>. Its main objective is to create "artificial variety" in your images so that the model is more robust and does not memorize the exact photos from training.</p>
<p>Here is the breakdown of each part:</p>
<ol>
    <li>
    models.Sequential([...])
    <p>Creates a small sequence of steps that will be applied to each image just before entering the neural network. It's like a "customs" through which the photos pass to be transformed.</p>
    </li>
    <li>
    layers.RandomFlip("horizontal")
    <p><strong>What it does:</strong> Flips the image horizontally (like a mirror) randomly.</p>
    <p><strong>Why it helps:</strong> A dog is still a dog even if it's facing the other way. This helps the model not to rely on the direction the animal is facing.</p>
    </li>
    <li>
    layers.RandomRotation(0.1)
    <p><strong>What it does:</strong> Rotates the image randomly by up to <strong>10%</strong> (about 36 degrees) in either direction.</p>
    <p><strong>Why it helps:</strong> In real life, photos are not always perfectly level. This teaches the model to recognize the animal even if the camera is slightly tilted.</p>
    </li>
    <li>
    layers.RandomZoom(0.1)
    <p><strong>What it does:</strong> Applies a zoom (inward or outward) randomly of up to <strong>10%</strong>.</p>
    <p><strong>Why it helps:</strong> Helps the model recognize objects even if they appear closer or further away from the camera.</p>
    </li>
</ol>
<hr>
<p>What is all this for?</p>
<p><strong>Data Augmentation</strong> is the best tool against <strong>Overfitting</strong>. Without this, if all your dog photos had the dog on the right, the model might incorrectly learn that "dog = something on the right". By flipping, rotating, and zooming, you force the model to learn <strong>the real features</strong> (ears, snout, eyes) instead of just the pixel position.</p>
<p><strong>Technical note:</strong> These transformations only occur during training and are random in each epoch, so the model almost never sees the exact same image twice.</p>

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
<p>This is the architecture of your <strong>Convolutional Neural Network (CNN)</strong>. Here is where the "intelligence" that will learn to distinguish between dogs and cats is defined. It reads like an assembly line, where the image enters from the top and the result comes out from the bottom:</p>

<ol>
    <li>
    Input and Preprocessing
    <p><strong>layers.Input(shape=(160, 160, 3)):</strong> Defines the input size. 160x160 are the pixels and 3 indicates the color channels (Red, Green, Blue - RGB).</p>
    <p><strong>data_augmentation:</strong> Here you insert the block we explained earlier. Each image that enters will be randomly flipped or rotated before being analyzed.</p>
    <p><strong>layers.Rescaling(1./255):</strong> Digital images have pixel values from 0 to 255. Neural networks work better with small numbers; this line divides everything by 255 so that the values are between <strong>0 and 1</strong>.</p>
    </li>
    <li>
    Feature Extraction (Conventional Layers)
    <p>This is the "visual" part of the model:</p>
    <p><strong>layers.Conv2D(32, (3, 3), activation='relu'):</strong> It's a "filter" that scans the image looking for simple patterns (edges, lines). It creates 32 distinct feature maps.</p>
    <p><strong>layers.MaxPooling2D(2, 2):</strong> Reduces the image size by half. It keeps only the most important information so that the model is faster and doesn't get lost in irrelevant details.</p>
    <p><strong>layers.Conv2D(64, ...):</strong> Another filter, but now it looks for more complex patterns (curves, shapes of eyes or ears). Having 64 filters, it can "see" more details.</p>
    </li>
    <li>
    Classification (Dense Layers)
    <p>Here is where the model takes the visual information and makes a decision:</p>
    <p><strong>layers.Flatten():</strong> Converts the two-dimensional feature maps into a long single list of numbers (a flat vector).</p>
    <p><strong>layers.Dense(128, activation='relu'):</strong> A layer of 128 "neurons" that connect all the detected patterns to try to understand what the animal is.</p>
    <p><strong>layers.Dropout(0.5):</strong> It's a safety technique. It "turns off" 50% of the neurons randomly at each training step. This forces the model not to rely on a single neuron and to be more robust (prevents overfitting).</p>
    <p><strong>layers.Dense(1, activation='sigmoid'):</strong> The final neuron.</p>
    <p>It uses <strong>sigmoid</strong> because there are only two options. It will return a number between <strong>0 and 1</strong>. If the result is close to 0, the model thinks it's one class (e.g., cat); if it's close to 1, it thinks it's the other (e.g., dog).</p>    
    </li>
</ol>

<p>Flow summary:</p>
<p>1. <strong>The image enters</strong> → 2. <strong>It is transformed</strong> (augmentation) → 3. <strong>It is normalized</strong> (0-1) → 4. <strong>Shapes are detected</strong> (Conv2D) → 5. <strong>It is simplified</strong> (Pooling) → 6. <strong>Information is analyzed</strong> (Dense) → 7. <strong>A final result is given</strong>.</p>

```
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```
<p>This part of the code is the <strong>configuration phase</strong> of the model. Before starting to train with the data, you must tell the model "how" it should learn and "how" you will measure if it is doing it right.</p>
<p>Here is the breakdown of the three pillars of learning:</p>

<ol>
    <li>
        optimizer='adam' (The Brain)
        <p>The optimizer is the algorithm that updates the weights of the neural network to reduce the error.</p>
        <p><strong>Adam</strong> is the industry standard today (2025). It's very efficient because it automatically adjusts the "learning rate". If the model is far from the answer, it takes large steps; if it's close, it takes small steps to avoid overshooting.</p>
    </li>
    <li>
        loss='binary_crossentropy' (The Measuring Rule)
        <p>The "loss function" is how the model calculates <strong>how wrong it is</strong>.</p>
        <p><strong>binary_crossentropy</strong> is used specifically when you have a <strong>binary classification</strong> (only two options, like cat or dog).</p>
        <p>If the model is very sure that a photo is a cat and it turns out to be a dog, this function will give it a very high "penalty" to learn from the mistake drastically.</p>
    </li>
    <li>
        metrics=['accuracy'] (The Report)
        <p>Here you define what statistics you want to see while the model trains.</p>
        <p><strong>accuracy (Precision):</strong> It's the percentage of correct guesses. For example, if out of 100 images the model guesses 90 correctly, you will see a 0.90 on your screen during training. It's the easiest metric for us humans to understand if the model is working.</p>
    </li>
</ol>

<p>In summary, with this line you are telling the program:</p>
<p><em>"Use the Adam algorithm to improve, measure your errors with Binary Crossentropy, and
show me the percentage of correct guesses after each step."</em></p>

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
<p>This code snippet uses the TensorFlow/Keras library to train a neural network.</p>

<p>Here is the step-by-step breakdown:</p>

<ol>
    <li><p><strong>history =:</strong> The fit method returns an object called History. This object records the metrics (like precision and loss) obtained at the end of each epoch, allowing you to graph the model's performance later. You can check the official Keras documentation to see what data it stores.</p></li>
    <li><p><strong>model.fit(...):</strong> This is the main function to start training. It "fits" (adjusts) the model's parameters (weights and biases) so that it learns to predict correctly based on the provided data.</p></li>
    <li><p><strong>train_ds:</strong> Represents the <strong>training dataset</strong>. It is the source of information from which the model will learn the patterns. Generally, it is an object of type tf.data.Dataset.</p></li>
    <li><p><strong>validation_data=val_ds:</strong> Defines the <strong>validation dataset</strong>. At the end of each epoch, the model is tested with this data (which it has not seen during training) to check if it is generalizing well or if it is suffering from overfitting.</p></li>
    <li><p><strong>epochs=10:</strong> Indicates the <strong>number of epochs</strong>. An epoch is a complete pass of all the training data through the neural network. In this case, the process will be repeated 10 times.</p></li>
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
<p>This part of the code is responsible for <strong>result visualization</strong>. It uses the Matplotlib library to create a graph that allows you to visually evaluate if your model learned correctly or if it has problems.</p>
<p>Here is the step-by-step breakdown:</p>
<ol>
    <li>
        import matplotlib.pyplot as plt
        <p>Imports the standard Python drawing tool. It is assigned the alias plt so that it is faster to write.</p>
    </li>
    <li>
        plt.plot(history.history['accuracy'], ...)
        <p><strong>What does it do?:</strong> Draws the line of <strong>training precision</strong>.</p>
        <p><strong>Where does it come from:</strong> Remember that we saved the training in a variable called history. This variable contains a dictionary with the record of how well the model did in each of the 10 epochs.</p>
    </li>
    <li>
        plt.plot(history.history['val_accuracy'], ...)
        <p><strong>What does it do?:</strong> Draws the line of <strong>validation precision</strong>.</p>
        <p><strong>Why is it important:</strong> This is the key line. It shows how well the model performs with images that <strong>it did not use for training</strong>.</p>
    </li>
    <li>
        Graph Configuration (Titles and Labels)
        <p><strong>plt.title, plt.ylabel, plt.xlabel:</strong> Adds the main title ("Model precision") and the names to the axes (the vertical axis is the success percentage and the horizontal one is the epochs or iterations).</p>
        <p><strong>plt.legend():</strong> Shows the box that indicates which line color corresponds to "Training" and which to "Validation".</p>
    </li>
    <li>
        plt.show()
        <p>It's the final order that opens the window and projects the graph on your screen.</p>
    </li>
</ol>
<hr>
<p>How to interpret this graph in 2025?</p>
<p>When you run this, you will see two lines. Ideally, both should go up together. Here are two common scenarios:</p>

<ol>
    <li>
        <p><strong>Ideal Scenario:</strong> Both lines go up and end up close to each other (e.g., 90%). This means your model learned to identify dogs and cats in general.</p>
    </li>
    <li>
        <p><strong>Overfitting:</strong> The Training line goes up to 99% but the Validation one stays flat or starts to go down. This means the model <strong>memorized</strong> the training photos but does not know how to recognize new photos.</p>
    </li>
</ol>

```
# Load the model and test
import os
import numpy as np
from tensorflow.keras.preprocessing import image

# set the environment
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

# suppress low level warnings
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
<p>This section of the code is for <strong>Inference</strong>. It is used to apply the knowledge that the model acquired during training to new images.</p>
<p>Here is the step-by-step breakdown:</p>

<ol>
    <li>
        Environment Setup and Loading
        <p><strong>os.environ[...]:</strong> These lines configure how TensorFlow interacts with your hardware. The log level '2' is used to clean the console, hiding informative messages and leaving only serious errors[1].</p>
        <p><strong>tf.keras.models.load_model('...'):</strong> This is the key function. It loads the .keras file that you saved earlier (which contains the structure and the "weights" or memory of the model). You no longer need to train again; the model already knows what to look for [2].</p>
    </li>
    <li>
        Image Preparation (predict_animal)
        <p>Before the model can "see" a photo, it must go through a transformation process:</p>
        <p><strong>image.load_img(..., target_size=(160, 160)):</strong> Opens the photo and forces it to be 160x160 pixels, just like you trained the model.</p>
        <p><strong>img_to_array:</strong> Converts the colors of the image into a matrix of numbers that the computer understands.</p>
        <p><strong>np.expand_dims(..., axis=0):</strong> TensorFlow does not accept a single loose image; it expects a "batch". This line converts the image from a single piece to a "list of one image", giving it the shape (1, 160, 160, 3).</p>
    </li>
    <li>
        Prediction and Decision Logic
        <p><strong>model.predict(img_array):</strong> The model analyzes the pixels and returns a decimal number between 0 and 1.</p>
        <p></p>
        <p><strong>Result Interpretation:</strong> As we used a sigmoid activation at the end of the model, the result is a probability.</p>
        <p><strong>if prediction[0] > 0.5:</strong> If the number is greater than 0.5, the model is more sure it's a <strong>dog</strong>. The closer to 1, the higher the confidence.</p>
        <p><strong>else:</strong> If it's less than 0.5, the model determines it's a <strong>cat</strong>.</p>
    </li>
    <li>
        Confidence Calculation (score)
        <p>For the dog, the direct value is the confidence.</p>
        <p>For the cat, it is subtracted from 1 (e.g., if it comes out 0.1, the model has a 1 - 0.1 = 0.9 or 90% confidence that it is a cat).</p>
        <p><strong>{score:.2%}:</strong> It's a text format that converts the number (e.g., 0.854) into a readable percentage (85.40%).</p>
    </li>
</ol>