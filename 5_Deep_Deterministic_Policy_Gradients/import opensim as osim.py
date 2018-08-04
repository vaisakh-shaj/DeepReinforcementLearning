import opensim as osim
from osim.http.client import Client
from osim.env import ProstheticsEnv

import os
import time
from collections import deque
import pickle

from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.ddpg import DDPG
import baselines.common.tf_util as U

from baselines import logger
import numpy as np
import tensorflow as tf
from mpi4py import MPI
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from osim.env import ProstheticsEnv
import sys

# Settings
remote_base = "http://grader.crowdai.org:1729"
crowdai_token = "8592db9b224e4293d437776321861a32"

client = Client(remote_base)
critic = Critic(layer_norm=layer_norm)
actor = Actor(nb_actions, layer_norm=layer_norm)
memory=[]
agent = DDPG(actor, critic, env.observation_space.shape, env.action_space.shape)
# Create environment
observation = client.env_create(crowdai_token)


# IMPLEMENTATION OF YOUR CONTROLLER
# my_controller = ... (for example the one trained in keras_rl)
def my_controller():
	with U.make_session() as sess:
        # Prepare everything.
        agent.initialize(sess)
        sess.graph.finalize()

        agent.reset()
        filename="/home/vaisakhs_shaj/Desktop/MODEL/MODEL/tfSteps"+str(80000)+".model"
        saver.restore(sess,filename)
		action, q = agent.pi(obs, apply_noise=False, compute_Q=True)
		return action


while True:
    [observation, reward, done, info] = client.env_step(my_controller(observation), True)
    print(observation)
    if done:
        observation = client.env_reset()
        if not observation:
            break

#client.submit()