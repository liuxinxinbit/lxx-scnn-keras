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
from skimage import io

t = Tusimple(image_set='train')
scnn = SCNN()

print(t.segLabel_list[2])
image = cv2.imread(t.segLabel_list[0],cv2.IMREAD_GRAYSCALE)

# image = io.imread(t.segLabel_list[0])
# image = Image.open(t.segLabel_list[0]) 
# image.show()
image = tf.cast(image, tf.float32)
print(type(image))
plt.imshow(image.numpy())
plt.show()

