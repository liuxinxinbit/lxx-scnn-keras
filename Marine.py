import json
import os
import glob
import matplotlib.pyplot as plt 
from labelme import utils
import imgviz
import cv2
import numpy as np
from numpy import random
from sklearn.preprocessing import StandardScaler
import torch

class Marine():
    """
    """

    def __init__(self, path='../marine_data'):
        super(Marine, self).__init__()
        self.data_dir_path = path
        self.scaler = StandardScaler()
        self.createIndex()

    def createIndex(self):
        self.img_list = self.read_traindata_names()

    def read_traindata_names(self,):
        trainset=[]
        for i in range(12):
            find_dir = os.path.join(self.data_dir_path,str(i+1),'images')
            # 'marine_data/'+ str(i+1) + '/images/'
            files = self.find_target_file(find_dir,'.json')
            trainset+=files
        return trainset
    def json2data(self, json_file):
        data = json.load(open(json_file))
        imageData = data.get('imageData')
        img = utils.img_b64_to_arr(imageData)

        label_name_to_value = {'_background_': 0}
        for shape in sorted(data['shapes'], key=lambda x: x['label']):
            label_name = shape['label']
            if label_name in label_name_to_value:
               label_value = label_name_to_value[label_name]
            else:
               label_value = len(label_name_to_value)
               label_name_to_value[label_name] = label_value
        lbl, _ = utils.shapes_to_label(
           img.shape, data['shapes'], label_name_to_value
        )
        label_names = [None] * (max(label_name_to_value.values()) + 1)
        for name, value in label_name_to_value.items():
            label_names[value] = name

        lbl_viz = imgviz.label2rgb(
            label=lbl, img=imgviz.asgray(img), label_names=label_names, loc='rb'
        )
        return img,lbl,lbl_viz
    def find_target_file(self,find_dir,format_name):
        files= [os.path.join(find_dir,file) for file in os.listdir(find_dir) if file.endswith(format_name)]
        return files

    def random_crop_or_pad(self, image, truth, size=(448, 512)):
        assert image.shape[:2] == truth.shape[:2]

        if image.shape[0] > size[0]:
            crop_random_y = random.randint(0, image.shape[0] - size[0])
            image = image[crop_random_y:crop_random_y + size[0],:,:]
            truth = truth[crop_random_y:crop_random_y + size[0],:]
        else:
            zeros = np.zeros((size[0], image.shape[1], image.shape[2]), dtype=np.float32)
            zeros[:image.shape[0], :image.shape[1], :] = image                                          
            image = np.copy(zeros)
            zeros = np.zeros((size[0], truth.shape[1]), dtype=np.float32)
            zeros[:truth.shape[0], :truth.shape[1]] = truth
            truth = np.copy(zeros)

        if image.shape[1] > size[1]:
            crop_random_x = random.randint(0, image.shape[1] - size[1])
            image = image[:,crop_random_x:crop_random_x + size[1],:]
            truth = truth[:,crop_random_x:crop_random_x + size[1]]
        else:
            zeros = np.zeros((image.shape[0], size[1], image.shape[2]))
            zeros[:image.shape[0], :image.shape[1], :] = image
            image = np.copy(zeros)
            zeros = np.zeros((truth.shape[0], size[1]))
            zeros[:truth.shape[0], :truth.shape[1]] = truth
            truth = np.copy(zeros) 
        return image, truth
    def BatchGenerator(self,batch_size=8, image_size=(448, 512, 3), labels=3):
        while True:
            images = np.zeros((batch_size, image_size[0], image_size[1], image_size[2]))
            truths = np.zeros((batch_size, image_size[0], image_size[1], labels))
            for i in range(batch_size):
                random_line = random.choice(self.img_list)
                image,truth_mask,lbl_viz = self.json2data(random_line)
                truth_mask=truth_mask+1
                # print(np.max(truth_mask))
                image, truth = self.random_crop_or_pad(image, truth_mask, image_size)
                images[i,:,:,:] = self.scaler.fit_transform(image.astype(np.float32).reshape(-1, 1)).reshape(-1, image.shape[0], image.shape[1], image.shape[2])
                truths[i,:,:,:] = (np.arange(labels) == truth[...,None]-1).astype(int) # encode to one-hot-vector
            yield images, truths

    def __len__(self):
        return len(self.img_list)

