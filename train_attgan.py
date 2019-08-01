import numpy as np

from env import *
from DataLoader import DataLoader
from AttGan import AttGan


def train():
    dloader = DataLoader(BATCH_SIZE)
    dloader.build()

    model = AttGan(eta=ETA, num_att=dloader.num_att)
    model.build()

    for e in range(EPOCHS):
        for X_src, X_att_a, X_att_b in dloader.next_batch():
            loss = model.step()


if __name__ == "__main__":
    train()

