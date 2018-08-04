import opensim as osim
from osim.http.client import Client
from osim.env import ProstheticsEnv

import os
import time
from collections import deque
import pickle
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.memory import Memory
from baselines.ddpg.ddpg import DDPG
import baselines.common.tf_util as U

from baselines import logger
import numpy as np
import tensorflow as tf
from mpi4py import MPI
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from osim.env import ProstheticsEnv
import sys

env = ProstheticsEnv(visualize=False)
env.change_model(model = '3D', difficulty = 2, prosthetic = True)
layer_norm=True    
nb_actions=19    
memory = Memory(limit=int(1.5e6), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape)
critic = Critic(layer_norm=layer_norm)
actor = Actor(nb_actions, layer_norm=layer_norm)
agent = DDPG(actor, critic, memory, env.observation_space.shape, env.action_space.shape,
        gamma=0.99)
saver=tf.train.Saver()
# IMPLEMENTATION OF YOUR CONTROLLER
# my_controller = ... (for example the one trained in keras_rl)

sess=tf.InteractiveSession()
agent.initialize(sess)
sess.graph.finalize()
agent.reset()
filename="/home/vaisakhs_shaj/Desktop/MODEL/tfSteps"+str(10000)+".model"
saver.restore(sess,filename)
observation=env.reset()
#print([n.name for n in tf.get_default_graph().as_graph_def().node])
def my_controller(obs):
	action=agent.pi(obs, apply_noise=False, compute_Q=False)[0] 
	return action

tr=0
s=0
while True:
    [observation, reward, done, info] = env.step(my_controller(observation))
    print(reward,done)
    s=s+1
    tr=tr+reward
    sys.stdout.flush()
    if done:
        observation = env.reset()
        print(tr)
        print(s)
        break

#client.submit()