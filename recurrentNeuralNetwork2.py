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

# read content of file
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# length of text is the number of characters in it
print('Length of text: {} characters'.format(len(text)))

print(text[:250])

# Encoding
# Each unique character will be encoded as a different integer
vocab = sorted(set(text))

#Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
