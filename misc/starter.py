import sys
import time

import numpy as np

import gym
import universe

environment = gym.make('wob.mini.ClickButton-v0')
env = environment
environment.configure(remotes=1)

while True:
        time.sleep(10000)
