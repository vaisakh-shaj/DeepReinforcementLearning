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
 # Augmented environment from the L2R challenge
def dict_to_list(state_desc):
    res = []
    pelvis = None

    for body_part in ["pelvis", "head","torso","toes_l","toes_r","talus_l","talus_r"]:
        if body_part in ["toes_r","talus_r"]:
            res += [0] * 9
            continue
        cur = []
        cur += state_desc["body_pos"][body_part][0:2]
        cur += state_desc["body_vel"][body_part][0:2]
        cur += state_desc["body_acc"][body_part][0:2]
        cur += state_desc["body_pos_rot"][body_part][2:]
        cur += state_desc["body_vel_rot"][body_part][2:]
        cur += state_desc["body_acc_rot"][body_part][2:]
        if body_part == "pelvis":
            pelvis = cur
            res += cur[1:]
        else:
            cur_upd = cur
            cur_upd[:2] = [cur[i] - pelvis[i] for i in range(2)]
            cur_upd[6:7] = [cur[i] - pelvis[i] for i in range(6,7)]
            res += cur

    for joint in ["ankle_l","ankle_r","back","hip_l","hip_r","knee_l","knee_r"]:
        res += state_desc["joint_pos"][joint]
        res += state_desc["joint_vel"][joint]
        res += state_desc["joint_acc"][joint]

    for muscle in sorted(state_desc["muscles"].keys()):
        res += [state_desc["muscles"][muscle]["activation"]]
        res += [state_desc["muscles"][muscle]["fiber_length"]]
        res += [state_desc["muscles"][muscle]["fiber_velocity"]]

    cm_pos = [state_desc["misc"]["mass_center_pos"][i] - pelvis[i] for i in range(2)]
    res = res + cm_pos + state_desc["misc"]["mass_center_vel"] + state_desc["misc"]["mass_center_acc"]

    return res
# Settings
remote_base = "http://grader.crowdai.org:1729"
crowdai_token = "01342e360022c2def5c2cc04c5843381"

Client = Client(remote_base)

layer_norm=True    
nb_actions=19    
memory = Memory(limit=int(1.5e6), action_shape=(158,), observation_shape=(19,))
critic = Critic(layer_norm=layer_norm)
actor = Actor(nb_actions, layer_norm=layer_norm)
agent = DDPG(actor, critic, memory, (158,), (19,),
    gamma=0.99)
saver=tf.train.Saver()
# IMPLEMENTATION OF YOUR CONTROLLER
# my_controller = ... (for example the one trained in keras_rl)

sess=tf.InteractiveSession()
agent.initialize(sess)
sess.graph.finalize()
agent.reset()
filename="/home/vaisakhs_shaj/Desktop/MODEL/tfSteps"+str(30000)+".model"
saver.restore(sess,filename)
# Create environment
observation = Client.env_create(env_id="ProstheticsEnv",token=crowdai_token)

#print([n.name for n in tf.get_default_graph().as_graph_def().node])

def my_controller(obs):
	obs=np.array(dict_to_list(obs))
	action=agent.pi(obs, apply_noise=False, compute_Q=False)[0] 
	action=action.tolist()
	return action





while True:
	[observation, reward, done, info] = Client.env_step(my_controller(observation), True)

	if done:
	    observation = Client.env_reset()
	    if not observation:
	        break

Client.submit()
