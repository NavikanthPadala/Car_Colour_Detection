import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import cv2

from tensorflow import keras
from tensorflow.keras import layers
# from tensorflow.keras.models import Sequential

model = tf.keras.models.load_model('vechicles_colors_detection.keras')
img_height = 180
img_width = 180
data_dir = "C:\\Users\\Navikanth$\\Downloads\\car_color\\train"
class_names = sorted(os.listdir(data_dir))

vechicle_path = "car_0.jpg"
img = tf.keras.utils.load_img(
    vechicle_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

predicted_class = class_names[np.argmax(score)]

plt.figure()
plt.imshow(img)
plt.title("Predicted: {}, Confidence: {:.2f}%".format(predicted_class, 100 * np.max(score)))
# plt.title("Predicted: {}".format(predicted_class))
plt.axis("off")
plt.show()