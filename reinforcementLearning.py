# Terminology
# Environment:
# This is what the agent will explore (ie. a level in mario)

# Agent:
# Entity that is exploring the environment. The agent interacts and takes
# actions within the env (ie. mario)

# State:
# The agent is in what we call a "state", for ie. it includes the location

# Action:
# Any interaction between the agent and the env, it may change the state of the agent
# Even not doing nothing is considered an action

# Reward:
# Every action taken will result in a reward of some magnitude (+ or -)
# The goal of the agent will be to maximize its reward.

# Q-Learning
# We use a matrix which represents the agent expected reward
# The agent learns by exploring the env and observing the outcome/reward from each action
# It will gradually start relying more heavily on what he has previously learnt

# The formula used to update the Q-Table afer each action is as follow
#Q[state,action]=Q[state,action]+α∗(reward+γ∗max(Q[newState,:])−Q[state,action]) 
#α stands for the Learning Rate (How big the change each update has)
#γ stands for the Discount Factor (How much focus is put on the current and future reward)

import gym

env = gym.make('FrozenLake-v0')
print(env.observation_space.n) #gets the number of states
print(env.action_space.n) #gets number of actions

env.reset()

action = env.action_space.sample() #gets a random action

new_state, reward, done, info = env.step(action) #take action
env.render() #renders the gui for the env

import numpy as np
import time

env = gym.make('FrozenLake-v0')
