from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
batch_size = 16
epochs = 10

dataset = '62_dataset'
train_dir = os.path.join(dataset, 'train')
valid_dir = os.path.join(dataset, 'validation')
img_size = (160, 60)


images = []
labels = []

v_images = []
v_labels = []

folders = os.listdir(train_dir)
for folder in folders:
    path = os.path.join(train_dir, folder)
    label = folder
    for file in os.listdir(path):
        impath = os.path.join(path, file)
        img = cv2.imread(impath)
        img = np.array(img, dtype=np.float32)
        img = img/255
        images.append(img)
        labels.append(label)

train_datagen = ImageDataGenerator(
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True
    )

train_generator = train_datagen.flow_from_directory(
    train_dir,
    img_size, 
    color_mode= 'rgb', 
    class_mode='categorical',
    batch_size= batch_size,
    shuffle=True
    )


x_train = np.array(images)
y_train = np.array(labels)



from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)

from sklearn.utils import shuffle
x_train, y_train = shuffle(x_train, y_train, random_state=42)

def get_model():
    

    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(3,3), padding= 'same', activation='relu', input_shape = (img_size[1], img_size[0], 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=16, kernel_size=(3,3), padding= 'same', activation='relu', input_shape = (img_size[1], img_size[0], 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Dropout(0.2))

    model.add(Conv2D(filters=32, kernel_size=(3,3), padding= 'same', activation='relu', input_shape = (img_size[1], img_size[0], 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=32, kernel_size=(3,3), padding= 'same', activation='relu', input_shape = (img_size[1], img_size[0], 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Dropout(0.2))

    model.add(Conv2D(filters=64, kernel_size=(3,3), padding= 'same', activation='relu', input_shape = (img_size[1], img_size[0], 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding= 'same', activation='relu', input_shape = (img_size[1], img_size[0], 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))


    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(units=512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(units=62, activation='softmax'))

    return model


model = get_model()
model.summary()
model.compile(
    optimizer=RMSprop(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics = ['accuracy'])
model.fit(
    x_train, y_train,
    validation_split=0.2,
    batch_size=batch_size,
    epochs=epochs)
model.save('vgg_like_v4.keras')




# model = Sequential()
#     model.add(Conv2D(filters=16, kernel_size=(3,3),  activation='relu', input_shape = (img_size[1], img_size[0], 3)))
#     model.add(BatchNormalization())
#     model.add(Conv2D(filters=16, kernel_size=(3,3),  activation='relu', input_shape = (img_size[1], img_size[0], 3)))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

#     model.add(Dropout(0.2))

#     model.add(Conv2D(filters=32, kernel_size=(3,3),  activation='relu', input_shape = (img_size[1], img_size[0], 3)))
#     model.add(BatchNormalization())
#     model.add(Conv2D(filters=32, kernel_size=(3,3),  activation='relu', input_shape = (img_size[1], img_size[0], 3)))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

#     model.add(Dropout(0.2))

#     model.add(Flatten())

#     model.add(Dense(units=512, activation='relu'))
#     model.add(Dropout(0.2))
#     model.add(Dense(units=62, activation='softmax'))