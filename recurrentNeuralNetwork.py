#This focusses on NLP (Natural Language Processing)

#How to encode data?
#1. Bag of words: 
# each word in a sentence is encoded with a number
# which is stored in a collection which does not keep track of the 
# order but does for the frequency\

vocab = {}
word_encoding = 1
def bag_of_words(text):
  global word_encoding

  words = text.lower().split(" ") #create a list of all of the words in the text, we do not care about grammar in this case
  bag = {} #stores all of the encodings and their frequency

  for word in words:
    if word in vocab:
      encoding = vocab[word] #get encoding from vocab
    else:
      vocab[word] = word_encoding
      encoding = word_encoding
      word_encoding += 1

    if encoding in bag:
      bag[encoding] += 1
    else:
      bag[encoding] = 1

  return bag

text = "this is a test to see if this test will work is is test a a"
bag = bag_of_words(text)
print(bag)
print(vocab)

#2. Integer Encoding:
# each word or character in a sentence is represented as a 
# unique int and the order is maintained

vocab = {}
word_encoding = 1

def one_hot_encoding(text):
  global word_encoding

  words = text.lower().split(" ")
  encoding = []

  for word in words:
    if word in vocab:
      code = vocab[word]
      encoding.append(code)
    else:
      vocab[word] = word_encoding
      encoding.append(word_encoding)
      word_encoding += 1

  return encoding

text = "this is a test to see if this test will work is is test a a"
encoding = one_hot_encoding(text)
print(encoding)
print(vocab)

#3. Word embeddings:
# this methods keeps track of the position and encodes similar words
# with similar labels
# each word is encoded as a dense vector that represents its context in the sentence

# we'll add a word embedding layer to the begging of our model

# Recurrent Neural Network work by passing words one by one (which is 
# not feed-forward where all data is fed at once like we use to do)
# the current word is processed in a combination with the output from the previous iteration

# Long Short-Term Memory (LSTM)
# Stores all the previous input as well as when they have been seen 

# Sentiment Analysis
# process of categorizing opinions expressed in a piece of text


from pickletools import optimize
from keras.datasets import imdb
from keras.preprocessing import sequence
import keras
import tensorflow as tf
import os
import numpy as np

VOCAB_SIZE = 88584

MAXLEN = 250
BATCH_SIZE = 64

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=VOCAB_SIZE)

# Lets look at one review
train_data[1]

# more preprocessing (triming if more than 250 words, adding zeros if less)

train_data = sequence.pad_sequences(train_data, MAXLEN)
test_data = sequence.pad_sequences(test_data, MAXLEN)