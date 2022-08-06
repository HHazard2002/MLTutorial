# RNN Play Generator
from pickletools import optimize
from syslog import LOG_SYSLOG
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

def text_to_int(text):
  return np.array([char2idx[c] for c in text])

text_as_int = text_to_int(text)

print("Text:", text[:13])
print("Encoded:", text_to_int(text[:13]))

def int_to_text(ints):
  try:
    ints = ints.numpy()
  except:
    pass
  return ''.join(idx2char[ints])

print(int_to_text(text_as_int[:13]))

seq_length = 100 # length of sequence for a training example
examples_per_epoch = len(text)//(seq_length+1)

# create training examples/targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

def split_input_target(chunk):
  input_text = chunk[:-1]
  target_text = chunk[1:]
  return input_text, target_text

dataset = sequences.map(split_input_target)

for x, y in dataset.take(2):
  print("\n\nExample\n")
  print("Input")
  print(int_to_text(x))
  print("\nOutput")
  print(int_to_text(y))

# making training batches
BATCH_SIZE = 64
VOCAB_SIZE = len(vocab)
EMBEDDING_DIM = 256
RNN_UNITS = 1024

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it does't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
