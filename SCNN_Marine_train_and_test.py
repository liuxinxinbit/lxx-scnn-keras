import torch
from Marine import Marine
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
from sklearn.preprocessing import StandardScaler

def random_crop_or_pad(image, size=(448, 512)):
    if image.shape[0] > size[0]:
        crop_random_y = random.randint(0, image.shape[0] - size[0])
        image = image[crop_random_y:crop_random_y + size[0], :, :]
    else:
        zeros = np.zeros(
            (size[0], image.shape[1], image.shape[2]), dtype=np.float32)
        zeros[:image.shape[0], :image.shape[1], :] = image
        image = np.copy(zeros)

    if image.shape[1] > size[1]:
        crop_random_x = random.randint(0, image.shape[1] - size[1])
        image = image[:, crop_random_x:crop_random_x + size[1], :]
    else:
        zeros = np.zeros((image.shape[0], size[1], image.shape[2]))
        zeros[:image.shape[0], :image.shape[1], :] = image
        image = np.copy(zeros)

    return image


image_size = (448, 512, 3)
dataset = Marine()
scnn = SCNN(image_size=image_size,nc=3)

scnn.set_generator(dataset.BatchGenerator(batch_size=6,image_size=image_size,labels=3))
scnn.load()
scnn.train(epochs=10, steps_per_epoch=250)
scnn.save()
scnn.load()
start = time.time()
for flag in range(500):
    print(str(flag).zfill(5))
    # image = np.float32(Image.open("../marine_data/11/images/"+str(flag+1).zfill(5)+".jpg"))
    image = cv2.imread("../marine_data/11/images/"+str(flag+1).zfill(5)+".jpg")
    train_image = random_crop_or_pad(image)
    train_image = scnn.transform(torch.from_numpy(train_image)).numpy()
    # train_image = StandardScaler().fit_transform(image.astype(np.float32).reshape(-1, 1)).reshape(-1, image.shape[0], image.shape[1], image.shape[2])[0,:,:,:]
    plt.subplot(1, 2, 1)
    
    plt.imshow(image)
    prediction = scnn.predict(train_image)
    print(prediction.shape)
    result = np.argmax(prediction[0,:,:, :],-1)
    print(result.shape)
    plt.subplot(1, 2, 2)
    plt.imshow(result)
    plt.pause(0.01)
    plt.clf()