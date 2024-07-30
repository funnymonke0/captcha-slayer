from keras.saving import load_model

import os
import cv2
import numpy as np

img_size = (160, 60)

batch_size = 16
epochs = 10

dataset = '62_dataset'
train_dir = os.path.join(dataset, 'train')
valid_dir = os.path.join(dataset, 'validation')

v_images = []
v_labels = []

folders = os.listdir(valid_dir)
for folder in folders:
    path = os.path.join(valid_dir, folder)
    label = folder
    for file in os.listdir(path):
        impath = os.path.join(path, file)
        img = cv2.imread(impath)
        img = np.array(img, dtype=np.float32)
        img = img/255
        v_images.append(img)
        v_labels.append(label)
x_valid = np.array(v_images)
y_valid = np.array(v_labels)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_valid = le.fit_transform(y_valid)

from sklearn.utils import shuffle
x_valid, y_valid = shuffle(x_valid, y_valid, random_state=42)




model = load_model('vgg_like_v4.keras')
model.evaluate(x_valid, y_valid, batch_size=batch_size)

