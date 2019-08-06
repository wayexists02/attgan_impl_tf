import numpy as np

HEIGHT = 192
WIDTH = 192
BATCH_SIZE = 128
ETA = 3e-3
EPOCHS = 100
GP = 1e-5
GPU_INDEX = [0, 1]

ATT_INDEX = np.array([4, 15, 20])

# DATA_PATH = "./data_sample_"
# IMAGE_PATH = "./data_sample_/img_align_celeba/"
# ATT_PATH = "./data_sample_/list_attr_celeba.txt"

DATA_PATH = "./data"
IMAGE_PATH = "./data/img_align_celeba/"
ATT_PATH = "./data/list_attr_celeba.txt"
TEST_IMAGE = "./data/img_align_celeba/000001.jpg"
