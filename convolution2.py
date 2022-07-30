import os
from attr import validate
import numpy as np
import matplotlib.pylot as plt
import tensorflow as tf
keras = tf.keras

import tensorflow_dataset as tfds
tfds.disable_progress_bar()

#split the data manually into 80% training, 10% testing, 10% validation
(raw_train, raw_validation, raw_test), metadata = tfds.load(
  'cats_vs_dog',
  split=['train[:80%]', 'test[80%:90%]', 'validation[90%:]'],
  with_info=True,
  as_supervised=True)

get_label_name = metadata.features['label'].int2str #creates a function object that we can use to get the labels

#display 2 images from the dataset
for image, label in raw_train.take(5):
  plt.figure()
  plt.imshow(image)
  plt.title(get_label_name(label))

IMG_SIZE = 160 #all images will be resized to  160x160

def format_example(image, label):
  image = tf.cast(image, tf.float32)
  image = (image/127.5) - 1
  image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
  return image, label

