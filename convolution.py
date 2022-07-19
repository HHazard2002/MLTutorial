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

#CNN architecture
#often a stack of conv2D and maxpooling2D followed by dense layers
#the first two extract the features from the image then these features are flattened and fed to the other layers
#which determine the class based on the presence of features

#building the convolutional base
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

#layer 1, input shape of the data is 32, 32, 3, we process 32 filter of size 3x3
# we also apply relu to the output of each convolution operation

#layer 2, this layer perform max pooling which will reduce the size by 2

#other layers, conv2d number of filter is doubled because the size of the data has been reduced and thus requires less computational power

model.summary()

#depth of the image increases but spacial dimensions reduce drastically


#dense layers
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

#training
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=4, validation_data=(test_images, test_labels))
