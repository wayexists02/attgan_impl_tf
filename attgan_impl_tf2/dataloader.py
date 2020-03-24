import numpy as np
import matplotlib.pyplot as plt
import pickle
import cv2

HEIGHT = 224
WIDTH = 224
CHANNEL = 3
NUM_ATT = 40


class DataLoader():

    def __init__(self, train_valid_test, batch_size):
        assert train_valid_test == "train" or train_valid_test == "valid" or train_valid_test == "test"

        self.train_valid_test = train_valid_test
        self.batch_size = batch_size

        self.dataset = self._load_data()

        self.num_batches = int(np.ceil(len(self.dataset) / self.batch_size))

    def __len__(self):
        if hasattr(self, "num_batch"):
            return self.num_batches

        return None

    def next_batch(self):
        dataset = self.dataset.copy()
        np.random.shuffle(dataset)

        for b in range(self.num_batches):
            start = b*self.batch_size
            end = min((b+1)*self.batch_size, len(dataset))

            x_batch = np.zeros((end - start, HEIGHT, WIDTH, CHANNEL))
            att_a_batch = np.zeros((end - start, NUM_ATT))
            att_b_batch = None

            for i in range(start, end):
                filepath, att = dataset[i]
                img = self._load_image(filepath)
                
                x_batch[i - start] = img
                att_a_batch[i - start] = list(map(lambda item: 1.0 if item == 1 else 0.0, att))
                
            att_b_batch = self._generate_attribute(att_a_batch)

            yield x_batch.astype(np.float32), att_a_batch.astype(np.float32), att_b_batch.astype(np.float32)

    def _load_data(self):
        with open(f"data/celeba/{self.train_valid_test}.bin", "rb") as f:
            dataset = pickle.load(f)

        return dataset

    def _load_image(self, path):
        img = plt.imread(path)
        img = cv2.resize(img, dsize=(HEIGHT, WIDTH))

        img = (img.astype(np.float32) - 128) / 256
        # img = img.astype(np.float32) / 256

        return img
    
    def _generate_attribute(self, a):
        b = a.copy()
        np.random.shuffle(b)
        return b
