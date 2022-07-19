#Dense and convolutional neural network are pretty similar, the difference is that
#dense search for general patterns while convolutional search for local patterns

#padding is used when we want the output of a filter to be the same size as the input
#strides represent how many rows/cols we will move the filter each time

#pooling reduce the size of the input by keeping the max, min, average
#usually used with a window of 2x2 and a stide of 2 which reduces the size by 2

#the dataset is composed of 60,000 32x32 color images with 6000 of each class

%tensorflow_version 2.x
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

#Load and split the dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

#Normalize pixel values to be between 0 and 1
train_images, test_images = train_images/255.0, test_images/255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#Let's look at one image
IMG_INDEX = 8

plt.imshow(train_images[IMG_INDEX], cmap=plt.cm.binary)
plt.xlabel(class_names[train_labels[IMG_INDEX][0]])
plt.show()