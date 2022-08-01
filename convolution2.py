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

#we can apply this function to all our data using map()
train = raw_train.map(format_example)
validatation = raw_validation.map(format_example)
test = raw_test.map(format_example)

for image, label in train.take(2):
  plt.figure()
  plt.imshow(image)
  plt.title(get_label_name(label))

#finally we shuffle and batch the image
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validatation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

for img, label in raw_train.take(2):
  print("Original shape:", img.shape)

for img, label in train.take(2):
  print("New shape:", img.shape)

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

#create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(
  input_shape=IMG_SHAPE,
  include_top=False,
  weights='imagenet')

base_model.summary()