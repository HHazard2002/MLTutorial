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