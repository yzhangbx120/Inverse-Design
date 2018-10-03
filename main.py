import os
import sys

import numpy as np
import tensorflow as tf

import vae_clean as vae

import matplotlib.pyplot as plt


IMG_DIM_X = 64
IMG_DIM_Y = 256
LAT_DIM = 100

ARCHITECTURE = [IMG_DIM_X*IMG_DIM_Y,
                512, 512, 512, 512,
                LAT_DIM] 

HYPERPARAMS = {
    "batch_size": 128,
    "learning_rate": 1E-4,       # 5e-4
    "dropout": 0.9,              # 0.9
    "lambda_l2_reg": 0.,       # 1e-5
    "nonlinearity": tf.nn.elu,
    "squashing": tf.nn.sigmoid
}

MAX_ITER = 1001
MAX_EPOCHS = np.inf


def main(to_reload=None):
    v = vae.VAE(ARCHITECTURE, HYPERPARAMS)
    v.predict_second(100)

if __name__ == "__main__":
    tf.reset_default_graph()

    try:
        to_reload = sys.argv[1]
        main(to_reload=to_reload)
    except(IndexError):
        main()
