import os
import time
from collections import deque
import pickle

from baselines.ddpg.ddpg import DDPG
import baselines.common.tf_util as U

from baselines import logger
import numpy as np
import tensorflow as tf
from mpi4py import MPI
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from osim.env import ProstheticsEnv
import sys
'''
not_restore = ['fc1_voc12_c0', 'fc1_voc12_c1', 'fc1_voc12_c2', 'fc1_voc12_c3']
restore_var = [v for v in tf.all_variables() if v.name not in not_restore] # Keep only the variables, whose name is not in the not_restore list.
'''
load_memory=True
restore=True
def train(env, nb_epochs, nb_epoch_cycles, render_eval, reward_scale, render, param_noise, actor, critic,
    normalize_returns, normalize_observations, critic_l2_reg, actor_lr, critic_lr, action_noise,
    popart, gamma, clip_norm, nb_train_steps, nb_rollout_steps, nb_eval_steps, batch_size, memory,
    tau=0.01, eval_env=None, param_noise_adaption_interval=50):
    rank = MPI.COMM_WORLD.Get_rank()
    #print(np.abs(env.action_space.low))
    #print(np.abs(env.action_space.high))
    #assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.
    max_action = env.action_space.high
    print(env.action_space)
    print(env.observation_space)
    #logger.info('scaling actions by {} before executing in env'.format(max_action))
    if load_memory:
        memory=pickle.load(open("/home/vaisakhs_shaj/Desktop/BIG-DATA/memory1000000.pickle","rb"))


    agent = DDPG(actor, critic, memory, env.observation_space.shape, env.action_space.shape,
        gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
        batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
        actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
        reward_scale=reward_scale)
 
    

    '''
     # Set up logging stuff only for a single worker.
    if rank == 0:
        saver = tf.train.Saver()
    else:
        saver = None
    '''
    saver=tf.train.Saver()
    step = 0
    episode = 0
    eval_episode_rewards_history = deque(maxlen=100)
    episode_rewards_history = deque(maxlen=10)
    with U.make_session() as sess:
        # Prepare everything.
        agent.initialize(sess)
        sess.graph.finalize()

        agent.reset()
        if restore:
            filename="/home/vaisakhs_shaj/Desktop/MODEL/tfSteps"+str(120000)+".model"
            saver.restore(sess,filename)
        obs = env.reset()

        if eval_env is not None:
            eval_obs = eval_env.reset()
        tr=0
        s=0
        while True:
            action=agent.pi(obs, apply_noise=False, compute_Q=False)[0] 
            obs, r, done, info = env.step(action)
            tr=tr+r
            s=s+1
            print(r)
            if done:
                print(tr)
                obs=env.reset()
                tr=0
                print(s)
                break
