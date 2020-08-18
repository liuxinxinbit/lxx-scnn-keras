from random import randint
import tensorflow as tf

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Lambda, Layer, BatchNormalization, Activation, concatenate,\
    Add, LeakyReLU, Dropout, AvgPool2D, Dense,ReLU
from tensorflow.keras import backend as K
from tensorflow.keras.applications import VGG16
import os
import numpy as np
import cv2
import os 
import matplotlib.pyplot as plt 
from tensorflow import keras, optimizers
from tensorflow.python.ops.image_ops_impl import ResizeMethod
import Tusimple
from sklearn.preprocessing import StandardScaler


class SCNN:
    def __init__(self, print_summary=False, image_size=(352, 640, 3), net_origin=False,nc=5):
        self.IMG_HEIGHT = image_size[0]
        self.IMG_WIDTH = image_size[1]
        self.num_classes = nc
        self.data_dir = ''
        if net_origin:
            self.build_origin(print_summary=print_summary,image_size=image_size,num_classes=self.num_classes)
        else:
            self.build(print_summary=print_summary,image_size=image_size,num_classes=self.num_classes)
        self.scaler = StandardScaler()

    def my_loss_error(self, y_true, y_pred):
        self.bce = keras.losses.BinaryCrossentropy()
        # self.ce = keras.losses.categorical_crossentropy()
        prob_output_loss = keras.losses.categorical_crossentropy(y_true[0],y_pred[0])
        prob_output_loss = tf.reduce_mean(prob_output_loss)
        existence_loss = self.bce(y_true[1],y_pred[1])
        existence_loss = tf.reduce_mean(existence_loss)
        total_loss = prob_output_loss + 0.0 * existence_loss
        return total_loss
    def predict(self, image):
        return self.model.predict(np.array([image]))
    
    def save(self, file_path='model.h5'):
        self.model.save_weights(file_path)
        
    def load(self, file_path='model.h5'):
        self.model.load_weights(file_path)

    def set_generator(self,train_batch_generator):
        self.train_batch_generator = train_batch_generator

    def train(self, epochs=10, steps_per_epoch=50):
        self.model.fit(self.train_batch_generator, steps_per_epoch=steps_per_epoch, epochs=epochs)

    def build_conv2D_block(self, inputs, filters, kernel_size, strides,dilation_rate=(1, 1)):
        conv2d = Conv2D(filters = filters, kernel_size=kernel_size,strides=strides,dilation_rate=dilation_rate, padding='same', use_bias=True,bias_initializer='zeros')(inputs)
        conv2d = BatchNormalization()(conv2d)
        conv2d_output = Activation(LeakyReLU(alpha=0.1))(conv2d)
        return conv2d_output

    def build_conv2Dtranspose_block(self, inputs, filters, kernel_size, strides):
        conv2d = Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, use_bias=True,bias_initializer='zeros', padding='same')(inputs)
        conv2d = BatchNormalization()(conv2d)
        conv2d_deconv = Activation(LeakyReLU(alpha=0.1))(conv2d)
        return conv2d_deconv

    def build_DepthwiseConv2D_block(self, inputs, filters):
        Depthwiseconv2d = keras.layers.DepthwiseConv2D(filters, padding="same")(inputs)
        Depthwiseconv2d = BatchNormalization()(Depthwiseconv2d)
        Depthwiseconv2d = Activation(LeakyReLU(alpha=0.1))(Depthwiseconv2d)
        return Depthwiseconv2d

    def build_SeparableConv2D_block(self, inputs, filters,kernel_size,strides):
        Separableconv2d = keras.layers.SeparableConv2D(filters,kernel_size,strides, padding="same")(inputs)
        Separableconv2d = BatchNormalization()(Separableconv2d)
        Separableconv2d = Activation(LeakyReLU(alpha=0.1))(Separableconv2d)
        return Separableconv2d
    def space_cnn_part(self,input_data):
        # add message passing #
        # top to down #
        feature_list_new = []
        nB, H, W, C = input_data.get_shape().as_list()
        slices = [input_data[:, i:(i + 1), :, :] for i in range(H)]
        out = [slices[0]]
        for i in range(1, len(slices)):
            tem = Conv2D(filters=128, kernel_size=[1, 9], strides=1, padding='SAME')(out[i - 1])
            tem = ReLU()(tem)
            conv_6 = Add()([tem, slices[i]])
            out.append(conv_6)
        slices = out[::-1]

        out = [slices[0]]
        for i in range(1, len(slices)):
            tem = Conv2D(filters=128, kernel_size=[1, 9], strides=1, padding='SAME')(out[i - 1])
            tem = ReLU()(tem)
            conv_6 = Add()([tem, slices[i]])
            out.append(conv_6)
        slices = out[::-1]

        feature_list_new = K.stack(slices, axis=1)
        feature_list_new = K.squeeze(feature_list_new, axis=2)

        slices = [feature_list_new[:, :, i:(i + 1), :] for i in range(W)]
        out = [slices[0]]
        for i in range(1, len(slices)):
            tem = Conv2D(filters=128, kernel_size=[9, 1], strides=1, padding='SAME')(out[i - 1])
            tem = ReLU()(tem)
            conv_6 = Add()([tem, slices[i]])
            out.append(conv_6)
        slices = slices[::-1]

        out = [slices[0]]
        for i in range(1, len(slices)):
            tem = Conv2D(filters=128, kernel_size=[9, 1], strides=1, padding='SAME',)(out[i - 1])
            tem = ReLU()(tem)
            conv_6 = Add()([tem, slices[i]])
            out.append(conv_6)
        slices = out[::-1]
        feature_list_new = K.stack(slices, axis=2)
        feature_list_new = K.squeeze(feature_list_new, axis=3)
        return feature_list_new


    def build(self, print_summary=False, num_classes=5,image_size=(352, 640, 3)):
          
        input_tensor = keras.layers.Input(image_size)
        conv_0 = self.build_conv2D_block(input_tensor, filters=24, kernel_size=1, strides=1)
        conv_0 = self.build_conv2D_block(conv_0, filters=24, kernel_size=3, strides=1)

        conv_0 = self.build_conv2D_block(conv_0, filters=24, kernel_size=3, strides=1)
        conv_0 = self.build_conv2D_block(conv_0, filters=24, kernel_size=3, strides=1)
        
        # first conv layer
        conv_1 = self.build_conv2D_block(conv_0, filters=48, kernel_size=3, strides=2)
        conv_1 = self.build_conv2D_block(conv_1, filters=48, kernel_size=3, strides=1)

        conv_1 = self.build_conv2D_block(conv_1, filters=48, kernel_size=3, strides=1)
        conv_1 = self.build_conv2D_block(conv_1, filters=48, kernel_size=3, strides=1)
        conv_1 = self.build_conv2D_block(conv_1, filters=48, kernel_size=3, strides=1)
        # second conv layer
        conv_2 = self.build_conv2D_block(conv_1, filters=64, kernel_size=3, strides=2)
        conv_2 = self.build_conv2D_block(conv_2, filters=64, kernel_size=3, strides=1)

        conv_2 = self.build_conv2D_block(conv_2, filters=64, kernel_size=3, strides=1)
        conv_2 = self.build_conv2D_block(conv_2, filters=64, kernel_size=3, strides=1)
        conv_2 = self.build_conv2D_block(conv_2, filters=64, kernel_size=3, strides=1)
        # third conv layer
        conv_3 = self.build_conv2D_block(conv_2, filters=96, kernel_size=3, strides=2)
        conv_3 = self.build_conv2D_block(conv_3, filters=96, kernel_size=3, strides=1)

        conv_3 = self.build_conv2D_block(conv_3, filters=96, kernel_size=3, strides=1)
        conv_3 = self.build_conv2D_block(conv_3, filters=96, kernel_size=3, strides=1)
        conv_3 = self.build_conv2D_block(conv_3, filters=96, kernel_size=3, strides=1)
        # fourth conv layer
        conv_4 = self.build_conv2D_block(conv_3, filters=128, kernel_size=3, strides=2)
        conv_4 = self.build_conv2D_block(conv_4, filters=128, kernel_size=3, strides=1)

        conv_4 = self.build_conv2D_block(conv_4, filters=128, kernel_size=3, strides=1)
        conv_4 = self.build_conv2D_block(conv_4, filters=128, kernel_size=3, strides=1)
        conv_4 = self.build_conv2D_block(conv_4, filters=128, kernel_size=3, strides=1)
        # fifth conv layer
        conv_5 = self.build_conv2D_block(conv_4, filters=256, kernel_size=3, strides=1, dilation_rate=1)
        conv_5 = self.build_conv2D_block(conv_5, filters=256, kernel_size=3, strides=1, dilation_rate=1)

        conv_5 = self.build_conv2D_block(conv_5, filters=256, kernel_size=3, strides=1, dilation_rate=1)
        conv_5 = self.build_conv2D_block(conv_5, filters=256, kernel_size=3, strides=1, dilation_rate=1)
        conv_5 = self.build_conv2D_block(conv_5, filters=256, kernel_size=3, strides=1, dilation_rate=1)
        # added part of SCNN #
        conv_6_4 = self.build_conv2D_block(conv_5, filters=256, kernel_size=3, strides=1, dilation_rate=1)
        conv_6_5 = self.build_conv2D_block(conv_6_4, filters=128, kernel_size=1, strides=1)  # 8 x 36 x 100 x 128

        scnn_part = self.space_cnn_part(conv_6_5)



        #######################
        # conv2d_deconv5_1 = self.build_conv2D_block(conv_5,filters = 196,kernel_size=3,strides=1)
        # conv2d_deconv4   = self.build_conv2Dtranspose_block(conv2d_deconv5_1, filters=128, kernel_size=4, strides=2)

        Concat_concat4 = concatenate([scnn_part, conv_4], axis=-1)

        conv2d_deconv4_1 = self.build_conv2D_block(Concat_concat4,filters = 96,kernel_size=3,strides=1)
        conv2d_deconv3   = self.build_conv2Dtranspose_block(conv2d_deconv4_1, filters=96, kernel_size=4, strides=2)

        Concat_concat3 = concatenate([conv2d_deconv3 , conv2d_deconv3] , axis=-1)

        conv2d_deconv3_1 = self.build_conv2D_block(Concat_concat3,filters = 64,kernel_size=3,strides=1)
        conv2d_deconv2   = self.build_conv2Dtranspose_block(conv2d_deconv3_1, filters=64, kernel_size=4, strides=2)

        Concat_concat2 = concatenate([conv2d_deconv2 , conv_2] , axis=-1)

        conv2d_deconv2_1 = self.build_conv2D_block(Concat_concat2,filters = 32,kernel_size=3,strides=1)
        conv2d_deconv1   = self.build_conv2Dtranspose_block(conv2d_deconv2_1, filters=32, kernel_size=4, strides=2)

        Concat_concat1 = concatenate([conv2d_deconv1 , conv_1] , axis=-1)

        conv2d_deconv1_1 = self.build_conv2D_block(Concat_concat1,filters = 16,kernel_size=3,strides=1)
        conv2d_deconv0  = self.build_conv2Dtranspose_block(conv2d_deconv1_1, filters=128, kernel_size=4, strides=2)

        if num_classes==1:
            ret_prob_output = Conv2DTranspose(filters=num_classes, kernel_size=1, strides=1, activation='sigmoid', padding='same', name='ctg_out_1')(conv2d_deconv0)
        else:
            ret_prob_output = Conv2DTranspose(filters=num_classes, kernel_size=1, strides=1, activation='softmax', padding='same', name='ctg_out_1')(conv2d_deconv0)

        ### add lane existence prediction branch ###
        # spatial softmax #
        # features = ret_prob_output  # N x H x W x C
        # softmax = Activation('softmax')(features)
        # avg_pool = AvgPool2D(strides=2)(softmax)
        features = self.build_conv2D_block(ret_prob_output,filters = num_classes,kernel_size=1,strides=2)
        _, H, W, C = features.get_shape().as_list()
        reshape_output = tf.reshape(features, [-1, H * W * C])
        fc_output = Dense(128)(reshape_output)
        relu_output = Activation('relu')(fc_output)
        existence_output = Dense(4)(relu_output)
        existence_output = Activation('softmax',name='ctg_out_2')(existence_output)
        self.model = Model(inputs=input_tensor, outputs=[ret_prob_output,existence_output])
        # print(self.model.summary())
        adam = optimizers.Adam(lr=0.001)
        sgd = optimizers.SGD(lr=0.001)
        
        if num_classes==1:
            self.model.compile(optimizer=sgd, loss="binary_crossentropy", metrics=['accuracy'])
        else:
            self.model.compile(optimizer=sgd, loss=self.my_loss_error, metrics=['accuracy'])
            
    #         self.model.compile(optimizer=sgd,   loss={
    #      'ctg_out_1': 'categorical_crossentropy',
    #      'ctg_out_2': 'binary_crossentropy'},
    #    loss_weights={
    #      'ctg_out_1': 1.,
    #      'ctg_out_2': 0.2,
    #    }, metrics=['accuracy', 'mse'])

        
    def build_origin(self, print_summary=False, num_classes=5,image_size=(352, 640, 3)):
          
        input_tensor = keras.layers.Input(image_size)
        conv_0 = self.build_conv2D_block(input_tensor, filters=24, kernel_size=1, strides=1)
        # conv stage 1
        conv_1 = self.build_conv2D_block(conv_0, filters=64, kernel_size=3, strides=1)
        conv_1 = self.build_conv2D_block(conv_1, filters=64, kernel_size=3, strides=1)

        # pool stage 1
        pool1 = MaxPooling2D()(conv_1)
        # conv stage 2
        conv_2 = self.build_conv2D_block(pool1, filters=128, kernel_size=3, strides=1)
        conv_2 = self.build_conv2D_block(conv_2, filters=128, kernel_size=3, strides=1)

        # pool stage 2
        pool2 = MaxPooling2D()(conv_2)
        # conv stage 3
        conv_3 = self.build_conv2D_block(pool2, filters=256, kernel_size=3, strides=1)
        conv_3 = self.build_conv2D_block(conv_3, filters=256, kernel_size=3, strides=1)
        conv_3 = self.build_conv2D_block(conv_3, filters=256, kernel_size=3, strides=1)
        
        # pool stage 3
        pool3 = MaxPooling2D()(conv_3)
        # conv stage 4
        conv_4 = self.build_conv2D_block(pool3, filters=512, kernel_size=3, strides=1)
        conv_4 = self.build_conv2D_block(conv_4, filters=512, kernel_size=3, strides=1)

        conv_4 = self.build_conv2D_block(conv_4, filters=512, kernel_size=3, strides=1)
        # pool4 = MaxPooling2D()(conv_4)
        ### add dilated convolution ###
        # conv stage 5_1
        conv_5 = self.build_conv2D_block(conv_4, filters=512, kernel_size=3, strides=1, dilation_rate=2)
        conv_5 = self.build_conv2D_block(conv_5, filters=512, kernel_size=3, strides=1, dilation_rate=2)
        conv_5 = self.build_conv2D_block(conv_5, filters=512, kernel_size=3, strides=1, dilation_rate=2)

        # added part of SCNN #
        conv_6_4 = self.build_conv2D_block(conv_5, filters=1024, kernel_size=3, strides=1, dilation_rate=4)
        conv_6_5 = self.build_conv2D_block(conv_6_4, filters=128, kernel_size=1, strides=1)  # 8 x 36 x 100 x 128

        # add message passing #
        # top to down #

        feature_list_new = self.space_cnn_part(conv_6_5)

        #######################
        dropout_output = Dropout(0.9)(feature_list_new)
        conv_output = K.resize_images(dropout_output, height_factor=self.IMG_HEIGHT //
                                          dropout_output.shape[1], width_factor=self.IMG_WIDTH//dropout_output.shape[2], data_format="channels_last")
        ret_prob_output = Conv2D(filters=num_classes,
                             kernel_size=1,activation='softmax', name='ctg_out_1')(conv_output)

        ### add lane existence prediction branch ###
        # spatial softmax #
        features = ret_prob_output  # N x H x W x C
        softmax = Activation('softmax')(features)
        avg_pool = AvgPool2D(strides=2)(softmax)
        _, H, W, C = avg_pool.get_shape().as_list()
        reshape_output = tf.reshape(avg_pool, [-1, H * W * C])
        fc_output = Dense(128)(reshape_output)
        relu_output = ReLU(max_value=6)(fc_output)
        existence_output = Dense(4,name='ctg_out_2')(relu_output)


        self.model = Model(inputs=input_tensor, outputs=[ret_prob_output,existence_output])
        # print(self.model.summary())
        adam = optimizers.Adam(lr=0.001)
        sgd = optimizers.SGD(lr=0.001)
        
        if num_classes==1:
            self.model.compile(optimizer=sgd, loss="binary_crossentropy", metrics=['accuracy'])
        else:
            self.model.compile(optimizer=sgd,   loss={
         'ctg_out_1': 'categorical_crossentropy',
         'ctg_out_2': 'binary_crossentropy'},
       loss_weights={
         'ctg_out_1': 1.,
         'ctg_out_2': 0.2,
       }, metrics=['accuracy', 'mse'])
