import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Lambda, LSTM, Conv2D, BatchNormalization, MaxPooling2D, Dense, Bidirectional
import keras.backend as K

from keras.callbacks import ModelCheckpoint
import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt


import string
class CaptchaSlayer2():
    batch_size = 16
    epochs = 10
    img_size = (128, 32)
    input_shape = (img_size[1], img_size[0], 1)
    char_list = string.ascii_letters+string.digits
    


    def __init__(self, dataset):
        self.dataset = dataset
        self.train_dir = os.path.join(self.dataset, 'train')
        self.valid_dir = os.path.join(self.dataset, 'validation')
        self.gen_arch()
        


    def convert(self, text):
        text = list(text)
        encoded = []
        for letter in text:
            encoded.append(self.char_list.index(letter))
        return encoded
    
    def preprocess(self):
        images = []
        labels = []

        

        valid_images = []
        valid_labels = []


        
        def parse(args):
            path, images, labels = args
            for filename in os.listdir(path):
                impath = os.path.join(path, filename)
                img = cv2.imread(impath)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = np.array(img, dtype=np.float32)
                img = img/255

                txt = filename.removesuffix('.png')

                images.append(img)
                labels.append(self.convert(txt))
            return images, labels

        path = self.train_dir
        images, labels = parse([path, images, labels])
        path = self.valid_dir
        valid_images, valid_labels = parse([path, valid_images, valid_labels])




        self.x_train = [images, labels, 31, 5]
        self.y_train = np.zeros(len(images))
        self.x_valid = [valid_images, valid_labels, 31, 5]
        self.y_valid = np.zeros(len(valid_images))
        

    # def internet_model(self):
    #     inputs = Input(shape=(32,128,1))
 
    #     # convolution layer with kernel size (3,3)
    #     conv_1 = Conv2D(64, (3,3), activation = 'relu', padding='same')(inputs)
    #     # poolig layer with kernel size (2,2)
    #     pool_1 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv_1)
        
    #     conv_2 = Conv2D(128, (3,3), activation = 'relu', padding='same')(pool_1)
    #     pool_2 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv_2)
        
    #     conv_3 = Conv2D(256, (3,3), activation = 'relu', padding='same')(pool_2)
        
    #     conv_4 = Conv2D(256, (3,3), activation = 'relu', padding='same')(conv_3)
    #     # poolig layer with kernel size (2,1)
    #     pool_4 = MaxPooling2D(pool_size=(2, 1))(conv_4)
        
    #     conv_5 = Conv2D(512, (3,3), activation = 'relu', padding='same')(pool_4)
    #     # Batch normalization layer
    #     batch_norm_5 = BatchNormalization()(conv_5)
        
    #     conv_6 = Conv2D(512, (3,3), activation = 'relu', padding='same')(batch_norm_5)
    #     batch_norm_6 = BatchNormalization()(conv_6)
    #     pool_6 = MaxPooling2D(pool_size=(2, 1))(batch_norm_6)
        
    #     conv_7 = Conv2D(512, (2,2), activation = 'relu')(pool_6)
        
    #     squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)
    #     # bidirectional LSTM layers with units=128
    #     blstm_1 = Bidirectional(LSTM(128, return_sequences=True, dropout = 0.2))(squeezed)
    #     blstm_2 = Bidirectional(LSTM(128, return_sequences=True, dropout = 0.2))(blstm_1)
        
    #     outputs = Dense(len(self.char_list)+1, activation = 'softmax')(blstm_2)

    #     # model to be used at test time
    #     self.act_model = Model(inputs, outputs)


    def gen_arch(self):
        inputs = Input(shape=self.input_shape)

        conv_1 = (Conv2D(filters=64, kernel_size=(3,3), padding= 'same', activation='relu'))(inputs)
        batch_norm_1 = (BatchNormalization())(conv_1)
        conv_2 = (Conv2D(filters=64, kernel_size=(3,3), padding= 'same', activation='relu'))(batch_norm_1)
        batch_norm_2 = (BatchNormalization())(conv_2)
        pool_1 = (MaxPooling2D(pool_size=(2, 2), strides=2))(batch_norm_2)

        # (Dropout(0.2))

        conv_3 = (Conv2D(filters=128, kernel_size=(3,3), padding= 'same', activation='relu'))(pool_1)
        batch_norm_3 = (BatchNormalization())(conv_3)
        conv_4 = (Conv2D(filters=128, kernel_size=(3,3), padding= 'same', activation='relu'))(batch_norm_3)
        batch_norm_4 = (BatchNormalization())(conv_4)
        pool_2 = (MaxPooling2D(pool_size=(2, 2), strides=2))(batch_norm_4)

        # (Dropout(0.2))

        conv_5 = (Conv2D(filters=256, kernel_size=(3,3), padding= 'same', activation='relu'))(pool_2)
        batch_norm_5 = (BatchNormalization())(conv_5)
        conv_6 = (Conv2D(filters=256, kernel_size=(3,3), padding= 'same', activation='relu'))(batch_norm_5)
        batch_norm_6 = (BatchNormalization())(conv_6)
        pool_3 = (MaxPooling2D(pool_size=(2, 2), strides=2))(batch_norm_6)

        conv_7 = (Conv2D(filters=256, kernel_size=(3,3), padding= 'same', activation='relu'))(pool_3)
        batch_norm_7 = (BatchNormalization())(conv_7)
        conv_8 = (Conv2D(filters=256, kernel_size=(3,3), padding= 'same', activation='relu'))(batch_norm_7)
        batch_norm_8 = (BatchNormalization())(conv_8)
        pool_4 = (MaxPooling2D(pool_size=(2, 2), strides=2))(batch_norm_8)

        conv_9 = Conv2D(filters=512, kernel_size=(2, 2), activation='relu')(pool_4)
        # (Dropout(0.2))

        
        squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_9)

        blstm_1 = Bidirectional(LSTM(128, return_sequences=True, dropout = 0.2))(squeezed)
        blstm_2 = Bidirectional(LSTM(128, return_sequences=True, dropout = 0.2))(blstm_1)

        # (Dense(units=512, activation='relu'))
        # (BatchNormalization())
        # (Dropout(0.2))

        outputs = Dense(units=len(self.char_list)+1, activation = 'softmax')(blstm_2)
        self.architecture = Model(inputs, outputs)
        

        
    def summary(self):
        self.architecture.summary()
        
    
    
    # def prep_ctc_loss(self):
    #     arch = self.architecture

    #     labels = Input(name='labels', shape=[5], dtype='float32')
    #     input_length = Input(name='input_length', shape=[1], dtype='int64')
    #     label_length = Input(name='label_length', shape=[1], dtype='int64')

    #     def ctc_lambda(args):
    #         y_pred, labels, input_length, label_length = args

    #         return K.ctc_batch_cost(y_true=labels, y_pred=y_pred, input_length=input_length, label_length=label_length)
        
    #     final_out = Lambda(ctc_lambda, output_shape=(1,), name = 'ctc_loss')([arch.outputs, labels, input_length, label_length])
    #     self.loss_model = Model(inputs=[arch.inputs, labels, input_length, label_length], outputs= final_out)
    

        
    def train(self):

        def ctc_loss(y_true, y_pred):
            input_length = 31
            label_length = 5
            return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)

        final_model = self.architecture
        final_model.compile(
            optimizer='adam',
            loss=ctc_loss)
        
        save_path = 'captchaslayer_v2.1.keras'

        checkpoint_1 = ModelCheckpoint(save_path, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
        callbacks = [checkpoint_1]

        final_model.fit(
            x=self.x_train,
            y=self.y_train,
            validation_data=(self.x_valid, self.y_valid),
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=1,
            callbacks=callbacks)


model = CaptchaSlayer2('general_dataset')
model.summary()
model.preprocess()
# model.prep_ctc_loss()
model.train()
