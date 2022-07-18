import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.compat.v2.feature_column as fc

#load dataset

dftrain = pd.read_csv('https://storage.googleapis.com/tf_datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf_datasets/titanic/eval.csv')
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

#show first 5 items in our dataframe

dftrain.head()

#show statistical description of data

dftrain.describe()

dftrain.shape

#few functions to get graph representation

dftrain.age.hist(bins = 20)
dftrain.sex.value_counts().plot(kind='barh')
dftrain['class'].value_counts().plot(kind='barh')
pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').sex_xlabel('% survive')

#Let's convert the categorical data into numerical one

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'face']


feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique() #gets a list of all unique values from given feature column
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))
for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

print(feature_columns)

#We need to create an input function to define how our dataset will be converted into batches at each epoch

def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_fn():
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df)) #creates tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000) #randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs) #split dataset into batches
    return ds #return a batch of the dataset
  return input_fn #return a function object for use

train_input_fn = make_input_fn(dftrain, y_train) #here we'll call the input_fn to get a dataset object we can feed to the model
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

#creation of a linear estimator to utilize the linear regression algorithm

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
#we create a linear estimator by passing the feature columns created earlier

#first training of the model
linear_est.train(train_input_fn) #train
result = linear_est.evaluate(eval_input_fn) #get model metrics/stats by testing on testing data
print(result['accuracy']) #the result variable is simply a dict of stats about our model

x = [1, 2, 2.5, 3, 4]
y = [1, 4, 7, 9, 15]
plt.plot(x, y, 'ro')
plt.axis([0, 6, 0, 20])
