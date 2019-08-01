import numpy as np
import os
import cv2

from env import *


class DataLoader():

    def __init__(self, batch_size):
        self.batch_size = batch_size

        self.data = []
        self.att_names = []
        self.num_att = 0
        self.num_images = 0
        self.num_batches = 0

    def __len__(self):
        return self.num_batches

    def build(self):
        self._read_file_list()
        self.num_batches = int(np.ceil(self.num_images / self.batch_size))

        print(f"Number of images: {self.num_images}")
        print(f"Number of attributes: {self.num_att}")
        print(f"Number of batches: {self.num_batches}")

    def _read_file_list(self):
        with open(ATT_PATH) as f:
            self.num_images = int(f.readline()) # num of images
            att_names = f.readline() # attribute names
            self.att_names = att_names.split()
            self.num_att = len(self.att_names)

            for line in f:
                splited = line.split()
                file_name = os.path.join(IMAGE_PATH, splited[0]).replace("\\", "/")
                attr = list(map(int, splited[1:]))

                self.data.append((file_name, attr))

    def next_batch(self):
        np.random.shuffle(self.data)
        
        for b in range(self.num_batches):
            start = b*self.batch_size
            end = min((b+1)*self.batch_size, self.num_images)

            X_src = np.zeros((end - start, HEIGHT, WIDTH, 3))
            X_att_a = np.zeros((end - start, self.num_att))
            X_att_b = np.zeros((end - start, self.num_att))

            for i in range(end - start):
                img_path = self.data[i + start][0]
                attr_list = self.data[i + start][1]
                
                try:
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, dsize=(HEIGHT, WIDTH)).astype(np.float32)
                except:
                    print(img_path)
                
                img = (img - 128) / 128
                X_src[i] = img
                X_att_a[i] = attr_list
                X_att_b[i, :] = -1
                X_att_b[i, np.random.randint(self.num_att, size=10)] = 1

            yield X_src, X_att_a, X_att_b

