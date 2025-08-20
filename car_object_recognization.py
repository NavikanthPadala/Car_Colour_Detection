import os
import cv2
import numpy as np
import requests

# URLs to download the model files
prototxt_url = 'https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt'
caffemodel_url = 'https://github.com/chuanqi305/MobileNet-SSD/blob/master/mobilenet_iter_73000.caffemodel?raw=true'


# Function to download files
def download_file(url, filename):
    if not os.path.exists(filename):
        response = requests.get(url)
        with open(filename, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded {filename}")
    else:
        print(f"{filename} already exists.")


# Download the deploy.prototxt file
download_file(prototxt_url, 'deploy.prototxt')

# Download the mobilenet_iter_73000.caffemodel file
download_file(caffemodel_url, 'mobilenet_iter_73000.caffemodel')

# Load the model
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

# Load the image
img = cv2.imread("C:\\Users\\Navikanth$\\Downloads\\mm.webp")
if img is None:
    print("Error: Image not found.")
    exit()
(h, w) = img.shape[:2]

# Prepare the image for the network
blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 0.007843, (300, 300), 127.5)
net.setInput(blob)

# Perform detection
detections = net.forward()

# Initialize a counter for image filenames
counter = 0

# Loop over the detections
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    # Filter out weak detections by ensuring the `confidence` is greater than a minimum threshold
    if confidence > 0.2:  # You can adjust the confidence threshold
        idx = int(detections[0, 0, i, 1])

        # The class label for "car" in the COCO dataset used by MobileNet SSD is 7
        if idx == 7:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Crop the detected car
            cropped_img = img[startY:endY, startX:endX]

            # Save the cropped image
            cv2.imwrite(f'car_{counter}.jpg', cropped_img)
            counter += 1

            # Draw a bounding box around the car
            cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
            print("Image cropped successfully")




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