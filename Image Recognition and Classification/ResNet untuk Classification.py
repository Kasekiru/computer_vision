#forum diskusi untuk image classification
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the pre-trained model
model = ResNet50(weights='imagenet')

# Load and preprocess the image
img_path = 'cat.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Predict the class of the image
predictions = model.predict(x)
print('Predicted:', decode_predictions(predictions, top=3)[0])
