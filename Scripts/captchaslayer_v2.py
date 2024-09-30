import tensorflow as tf
from keras.models import Model
from keras.models import load_model
from keras.layers import Input, Lambda, LSTM, Conv2D, BatchNormalization, MaxPooling2D, Dense, Bidirectional
import keras.backend as K

from keras.callbacks import ModelCheckpoint
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import string

class CaptchaSlayer2():
    batch_size = 16
    epochs = 10
    img_size = (128, 32)
    rgb = 3
    grayscale = 1
    images = None
    valid_images = None
    labels = None
    valid_labels = None
    history = None
    input_shape = (img_size[1], img_size[0], grayscale)
    char_list = string.ascii_letters+string.digits
    save_path = 'captchaslayer_v2.1.h5'
    


    def __init__(self, dataset):
        self.dataset = dataset
        try:
            self.train_dir = os.path.join(self.dataset, 'train')
            
        except Exception as s:
            print(f'No folder named "train": {s}')
        try:
            self.valid_dir = os.path.join(self.dataset, 'validation')
        except Exception as s:
            print(f'No folder named "validation": {s}')
        
        try:
            self.gen_arch()
        except Exception as s:
            print(f'Initialization error, check inputs: {s}')
        
        


    def convert(self, text):
        text = list(text)
        encoded = []
        try:
            for letter in text:
                encoded.append(self.char_list.index(letter))
        except Exception as s:
            print(f'Error during encoding: {s}')
        return encoded
    
    def preprocess(self):
        self.images = []
        self.labels = []

        

        self.valid_images = []
        self.valid_labels = []


        
        def flow_from_directory(args):
            path, self.images, self.labels = args
            try:
                for filename in os.listdir(path):
                    impath = os.path.join(path, filename)
                    img = cv2.imread(impath)
                    img = cv2.resize(img, dsize=self.img_size, interpolation = cv2.INTER_AREA) #INTER_CUBIC is better unless shrinking image
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    img = np.array(img, dtype=np.float32) / 255
                    
                    txt = filename.removesuffix('.png')

                    self.images.append(img)
                    self.labels.append(self.convert(txt))
            except Exception as s:
                print(f'Failed to load data: {s}')

            return self.images, self.labels

        
        path = self.train_dir
        self.images, self.labels = flow_from_directory([path, self.images, self.labels])


        path = self.valid_dir
        self.valid_images, self.valid_labels = flow_from_directory([path, self.valid_images, self.valid_labels])

        self.input_length = len(self.images)
        self.x_train = np.array(self.images)


        self.label_length = len(self.labels)
        self.y_train = np.array(self.labels)



        self.x_valid = np.array(self.valid_images)
        self.y_valid = np.array(self.valid_labels)


       
        

    def decode(self, testlabel):
        try:
            translate = []
            for index in testlabel:
                translate.append(self.char_list[index])
            return ''.join(translate)
        except Exception as s:
            print(f'Error during decoding: {s}')



    def gen_arch(self):
        inputs = Input(shape=self.input_shape)

        conv_1 = (Conv2D(filters=64, kernel_size=(3,3), padding= 'same', activation='relu'))(inputs)
        batch_norm_1 = (BatchNormalization())(conv_1)
        conv_2 = (Conv2D(filters=64, kernel_size=(3,3), padding= 'same', activation='relu'))(batch_norm_1)
        batch_norm_2 = (BatchNormalization())(conv_2)
        pool_1 = (MaxPooling2D(pool_size=(2, 2), strides=2))(batch_norm_2)



        conv_3 = (Conv2D(filters=128, kernel_size=(3,3), padding= 'same', activation='relu'))(pool_1)
        batch_norm_3 = (BatchNormalization())(conv_3)
        conv_4 = (Conv2D(filters=128, kernel_size=(3,3), padding= 'same', activation='relu'))(batch_norm_3)
        batch_norm_4 = (BatchNormalization())(conv_4)
        pool_2 = (MaxPooling2D(pool_size=(2, 2), strides=2))(batch_norm_4)



        conv_5 = (Conv2D(filters=256, kernel_size=(3,3), padding= 'same', activation='relu'))(pool_2)
        batch_norm_5 = (BatchNormalization())(conv_5)
        conv_6 = (Conv2D(filters=256, kernel_size=(3,3), padding= 'same', activation='relu'))(batch_norm_5)
        batch_norm_6 = (BatchNormalization())(conv_6)
        pool_3 = (MaxPooling2D(pool_size=(2, 1)))(batch_norm_6)

        conv_7 = (Conv2D(filters=512, kernel_size=(3,3), padding= 'same', activation='relu'))(pool_3)
        batch_norm_7 = (BatchNormalization())(conv_7)
        conv_8 = (Conv2D(filters=512, kernel_size=(3,3), padding= 'same', activation='relu'))(batch_norm_7)
        batch_norm_8 = (BatchNormalization())(conv_8)
        pool_4 = (MaxPooling2D(pool_size=(2, 1)))(batch_norm_8)

        conv_9 = Conv2D(filters=512, kernel_size=(2, 2), activation='relu')(pool_4)


        
        squeezed = Lambda(lambda x: K.squeeze(x, axis =1))(conv_9)

        blstm_1 = Bidirectional(LSTM(128, return_sequences=True, dropout = 0.2))(squeezed)
        blstm_2 = Bidirectional(LSTM(128, return_sequences=True, dropout = 0.2))(blstm_1)


        outputs = Dense(units=len(self.char_list)+1, activation = 'softmax')(blstm_2)
        self.architecture = Model(inputs, outputs)





        
    def summary(self):
        try:
            self.architecture.summary()
        except Exception as s:
            print(f'Architecture has not been initiated yet.')
        
    def ctc_loss(self, y_true, y_pred):
            batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
            input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
            label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

            input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
            label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
            return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)

    def train(self):

        
        try:
            final_model = self.architecture
            final_model.compile(
                optimizer='adam',
                loss=self.ctc_loss)
        except Exception as s:
            print(f'Architecture has not been initiated yet or error during compilation: {s}')
        

        checkpoint_1 = ModelCheckpoint(filepath=self.save_path, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
        callbacks = [checkpoint_1]
        try:
            self.history = final_model.fit(
                x=self.x_train,
                y=self.y_train,
                validation_data=(self.x_valid, self.y_valid),
                batch_size=self.batch_size,
                epochs=self.epochs,
                verbose=1,
                callbacks=callbacks)
        except Exception as s:
            print(f'Error when trying to train model: {s}')
        try:
            plt.plot(self.history.history['loss'])
            plt.plot(self.history.history['val_loss'])
            plt.title('loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            plt.show()
        except Exception as s:
            print(f'Error plotting model performance: {s}')
        
    def validate(self):
        try:
            model = load_model(self.save_path, custom_objects={'ctc_loss': self.ctc_loss})
        except Exception as s:
            print(f'Failed to load model: {s}')
        try:
            score = model.evaluate(self.x_valid, self.y_valid, verbose = 0)
            print(f'loss: {score}')
        except Exception as s:
            print(f'Exception during evaluation: {s}')
            
        try: 
            prediction = model.predict(self.x_valid)

            out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1], greedy=True)[0][0])
            
            correct = 0

            for index, pred in enumerate(out):
                answer = self.decode(self.y_valid[index])
                print(answer)
                pred = [j for i, j in enumerate(pred) if j != -1]
                pred = self.decode(pred)
                print(str(pred))

                if pred == answer:
                    correct+=1
            
            print(f"{correct}/{len(self.y_valid)}")
            accuracy = round(100*(correct/len(self.y_valid)), 2)
            print(f"{str(accuracy)}% accuracy")

            row = 8
            col = 8
            size = (row, col)
            fig = plt.figure(figsize=size)

            for index, image in enumerate(self.y_valid):
                fig.add_subplot(row, col, index+1)
                plt.imshow(image) 
                plt.axis('off')
                plt.title(self.valid_labels[index])
            plt.show()


        except Exception as s:
            print(f'Exception while getting model predictions: {s}')




model = CaptchaSlayer2('general_dataset')

# model.summary()
model.preprocess()

# model.train()
model.validate()





