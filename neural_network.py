from pickletools import optimize
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplolib.pylot as plt

#we use the MNIST fashion dataset which is included in keras (60,000 images for training and 10,000 for testing)
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data() #split the data into training and testing

train_images.shape
#we have 60,000 images made up of 28x28 pixels

#labels go from 0 to 9 thus we make an array of label
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#we do some pre-processing, as images are on the greyscale (0-255) we divide them by 255 to get values from 0 to 1
#smaller values are faster to process
train_images = train_images/255
test_images = test_images/255

#building the model
model = keras.Sequential([
  keras.layers.Flatten(input_shape=(28,28)), #input layer (1)
  keras.layers.Dense(128, activation='relu'), #hidden layer (2)
  keras.layers.Dense(10, activation='softmax') #output layer (3)
])

#layer 1, input layer which consists of 784 neurons
#the flatten means that our layer will reshape the (28x28) array
#into a vector of 784 neurons so that each pixel will be associated with a neuron

#layer 2, hidden layer, dense denotes that this layer will be fully connected to the previous layer,
#meaning that each neuron from both layers are connected with each other), has 128 neurons

#layer 3, output layer, dense layer, has 10 neurons which we'll look at to determine the output
#each neuron represent the proba of the image being one of the 10 classes
#the activation function softmax is used to calculate the proba distrib for each class
#this means that any value in this layer will be between 0 and 1

#compile the model
model.compile(
  optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy']
)
#we define the loss function, optimizer and metrics

#training the model
model.fit(train_images, train_labels, epochs=10)
#we pass the data, labels and epochs

#testing the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
print('Test accuracy: ', test_acc)

#getting predictions
predictions = model.predict(test_images)

#now we can compare test_label[0] and np.argmax(predictions[0])

#here is a func to verify the predictions
COLOR = 'white'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR

def predict(model, image, correct_label):
  class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
  prediction = model.predict(np.array([image]))
  predicted_class = class_names[np.argmax(prediction)]

  show_image(image, class_names[correct_label], predicted_class)

def show_image(img, label, guess):
  plt.figure()
  plt.imshow(img, cmap=plt.cm.binary)
  plt.title("Expected: " + label)
  plt.xlabel("Guess: " + guess)
  plt.colorbar()
  plt.grid(False)
  plt.show

def get_number():
  while True:
    num = input("Pick a number: ")
    if num.isdigit():
      num = int(num)
      if 0 <= num <= 1000:
        return int(num)
    else:
      print("Try again ...")
  
num = get_number()
image = test_images[num]
label = test_labels[num]
predict(model, image, label)


