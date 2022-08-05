# RNN Play Generator
from pickletools import optimize
from syslog import LOG_SYSLOG
from convolutional2 import BATCH_SIZE
from keras.preprocessing import sequence
import keras
import tensorflow as tf
import os
import numpy as np

path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

from google.colab import files
path_to_file = list(files.upload().keys())[0]