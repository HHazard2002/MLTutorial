import tenserflow_probability as tfp
import tensorflow as tf

tfd = tfp.distributions #making a shortcut for later on
initial_distribution = tfd.Categorical(probs=[0.2, 0.8])
transition_distribution = tfd.Categorical(probs=[[0.5, 0.5], [0.2, 0.8]])
observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])

#the loc argument represents the mean and the scale is the standart deviation

model = tfd.HiddenMarkovModel(
  initial_distribution=initial_distribution,
  transition_distribution=transition_distribution,
  observation_distribution=observation_distribution,
  num_steps=7
)
#number of steps is the number of day we want the prediction for. here it is for a full week

#we now get the expected temperatures on each day
mean = model.mean()

with tf.combat.v1.Session() as sess:
  print(mean.numpy())

#values should be 12 11.1 10.83 10.748999 10.71741 10.715222