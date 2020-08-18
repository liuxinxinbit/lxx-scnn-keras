from tensorflow.python.ops.image_ops_impl import ResizeMethod
from Tusimple import Tusimple
from SCNN import SCNN
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import random
import imgviz
import time
from tensorflow.keras import backend as K
import  tensorflow as tf

num_class = 5
image_size = (352, 640,3)
train_dataset = Tusimple(image_set = "train")
scnn = SCNN(image_size=image_size,nc=num_class,net_origin=False)

scnn.set_generator(train_dataset.BatchGenerator(labels=num_class))
 

scnn.train(epochs=5, steps_per_epoch=250)
scnn.save(file_path="tusimple_model.h5")

scnn.load(file_path="tusimple_model.h5")
start = time.time()
for flag in range(1):
    print(str(flag).zfill(5))
    image = cv2.imread("demo/demo.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)/255
    image = tf.image.resize(image, [352, 640],method=ResizeMethod.BICUBIC).numpy()
    # inputdata = scnn.scaler.fit_transform(image.astype(np.float32).reshape(-1, 1)).reshape(-1, image.shape[0], image.shape[1], image.shape[2])[0,:,:,:]
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    prediction = scnn.predict(image)[0]
    print(prediction.shape)
    result = np.argmax(prediction[0,:,:,:],-1)
    plt.subplot(1, 2, 2)
    plt.imshow(result)
    plt.pause(0.5)
    # plt.clf()
    plt.show()
