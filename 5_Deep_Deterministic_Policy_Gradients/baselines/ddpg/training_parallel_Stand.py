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
nproc=4
load_memory=True
restore=True
def make_env(seed):
    def _f():
        
        #if seed%2==0:
        env=ProstheticsEnv(visualize=False,integrator_accuracy = 3e-3)
        env.change_model(model = '2D', difficulty = 2, prosthetic = False, seed=seed)
        #else:
        
        #env=ProstheticsEnv(visualize=False,integrator_accuracy = 3e-3)
        #env.change_model(model = '3D', difficulty = 0, prosthetic = True, seed=seed)
        
        '''
        if seed>=4 and seed<6: 
            env=ProstheticsEnv(visualize=False,integrator_accuracy = 3e-3)
            env.change_model(model = '2D', difficulty = 1, prosthetic = True, seed=seed)
        
        if seed>=6 and seed<8:
            env=ProstheticsEnv(visualize=False,integrator_accuracy = 3e-3)
            env.change_model(model = '3D', difficulty = 0, prosthetic = True, seed=seed)
        if seed>=8 and seed<10:
            env=ProstheticsEnv(visualize=False,integrator_accuracy = 3e-3)
            env.change_model(model = '3D', difficulty = 0, prosthetic = True, seed=seed)
        if seed>=10 and seed<12 :
            env=ProstheticsEnv(visualize=False,integrator_accuracy = 3e-3)
            env.change_model(model = '3D', difficulty = 0, prosthetic = True, seed=seed)
        #env.seed(seed)
        '''
        return env
    return _f

def train(env, nb_epochs, nb_epoch_cycles, render_eval, reward_scale, render, param_noise, actor, critic,
    normalize_returns, normalize_observations, critic_l2_reg, actor_lr, critic_lr, action_noise,
    popart, gamma, clip_norm, nb_train_steps, nb_rollout_steps, nb_eval_steps, batch_size, memory,
    tau=0.01, eval_env=None, param_noise_adaption_interval=50):
    rank = MPI.COMM_WORLD.Get_rank()
    #print(np.abs(env.action_space.low))
    #print(np.abs(env.action_space.high))
    #assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.
    max_action = env.action_space.high

    logger.info('scaling actions by {} before executing in env'.format(max_action))
    if load_memory:
        memory=pickle.load(open("/home/vaisakhs_shaj/Desktop/BIG-DATA/memoryNorm300000.pickle","rb"))
        '''
        samps = memoryPrev.sample(batch_size=memoryPrev.nb_entries)
        print(len(samps['obs0'][1]))
        for i in range(memoryPrev.nb_entries):
            memory.append(samps['obs0'][i], samps['actions'][i], samps['rewards'][i], samps['obs1'][i],  samps['terminals1'][i])
        print("=============memory loaded================")
        '''
    agent = DDPG(actor, critic, memory, env.observation_space.shape, env.action_space.shape,
        gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
        batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
        actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
        reward_scale=reward_scale)
    logger.info('Using agent with the following configuration:')
    logger.info(str(agent.__dict__.items()))
    envs = [make_env(seed) for seed in range(nproc)]
    envs = SubprocVecEnv(envs)
    
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
            filename="/home/vaisakhs_shaj/Desktop/MODEL/normal/tfSteps"+str(25000)+".model"
            saver.restore(sess,filename)
            print("loaded!!!!!!!!!!!!!")
            #p=[v.name for v in tf.all_variables()]
            #print(p)
        
        obs = envs.reset()

        if eval_env is not None:
            eval_obs = eval_env.reset()
        done = False
        episode_reward = 0.
        episode_reward3 = 0.
        episode_step = 0
        episode_step3 = 0
        episodes = 0
        t = 0

        epoch = 0
        start_time = time.time()

        epoch_episode_rewards = []
        epoch_episode_steps = deque(maxlen=10)
        epoch_episode_steps3 = deque(maxlen=10)
        epoch_episode_eval_rewards = []
        epoch_episode_eval_steps = []
        epoch_start_time = time.time()
        epoch_actions = []
        epoch_qs = []
        epoch_episodes = 0
        learning_starts = 10000
        for epoch in range(nb_epochs):
            print("cycle-memory")
            print(max_action)
            for cycle in range(nb_epoch_cycles):
                print(cycle,"-",memory.nb_entries,end=" ")
                sys.stdout.flush()
                # Perform rollouts.
                for t_rollout in range(nb_rollout_steps):
                    # Predict next action.
                    action = np.stack([agent.pi(obs[i], apply_noise=True, compute_Q=False)[0] for i in range(nproc)])
                    q = np.stack([agent.pi(obs[i], apply_noise=True, compute_Q=True)[1] for i in range(nproc)])
                    # action, q = agent.pi(obs, apply_noise=True, compute_Q=True)
                    #assert action.shape == env.action_space.shape
                    #print(i)
                    # Execute next action in parallel.
                    if rank == 0 and render:
                        env.render()
                    #assert max_action.shape == action.shape
                    new_obs, r, done, info = envs.step(action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                    t += 1
                    if rank == 0 and render:
                        env.render()
                    #print(r)
                    #print(r[1])
                    sys.stdout.flush()
                    episode_reward += r[1]
                    #episode_reward3 += r[2]
                    episode_step += 1
                    #episode_step3 += 1
                    '''
                    if episode_step==300:
                        e=episode_step
                        re=episode_reward
                    if episode_step>300:
                        episode_step=e
                        episode_reward=re
                    '''
                    #print(episode_step)

                    book_keeping_obs=obs
                    obs = new_obs
                    #print(envs[1])
                    #print(episode_reward)
                    # Book-keeping in parallel.
                    epoch_actions.append(np.mean(action))
                    epoch_qs.append(np.mean(q))
                    for i in range(nproc):
                        agent.store_transition(book_keeping_obs[i], action[i], r[i], new_obs[i], done[i])
                        #print(done)
                        if done[i]:
                            # Episode done.
                            #print("====done====",episode_reward)
                            if i==1:
                                
                                epoch_episode_rewards.append(episode_reward)
                                #rint(epoch_episode_rewards)
                                #episode_rewards_history.append(episode_reward)
                                epoch_episode_steps.append(episode_step)
                                episode_reward = 0.
                                #episode_reward3 = 0
                                episode_step = 0
                                epoch_episodes += 1
                                episodes += 1
                            '''
                            if i==2:
                                
                                #epoch_episode_rewards.append(episode_reward3)
                                #rint(epoch_episode_rewards)
                                episode_rewards_history.append(episode_reward3)
                                epoch_episode_steps3.append(episode_step3)
                                episode_reward3 = 0
                                episode_step3 = 0
                            '''    

                            agent.reset()
                            temp = envs.reset()
                            obs[i]=temp[i]
                            
                            
                    
                

                    '''
                    Variables in TensorFlow only have values inside sessions.
                    Once the session is over, the variables are lost.
                    saver,save and saver .restore depends on session and has to be inside the 
                    session.
                    '''
                
                        

                   

                    # Train.
                    epoch_actor_losses = []
                    epoch_critic_losses = []
                    epoch_adaptive_distances = []
                    for t_train in range(nb_train_steps):
                        # Adapt param noise, if necessary.
                        if memory.nb_entries >= batch_size and t_train % param_noise_adaption_interval == 0:
                            distance = agent.adapt_param_noise()
                            epoch_adaptive_distances.append(distance)

                        cl, al = agent.train()
                        epoch_critic_losses.append(cl)
                        epoch_actor_losses.append(al)
                        agent.update_target_net()

                # Evaluate.
                eval_episode_rewards = []
                eval_qs = []
                if eval_env is not None:
                    eval_episode_reward = 0.
                    for t_rollout in range(nb_eval_steps):
                        eval_action, eval_q = agent.pi(eval_obs, apply_noise=False, compute_Q=True)
                        eval_obs, eval_r, eval_done, eval_info = eval_env.step(max_action * eval_action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                        if render_eval:
                            eval_env.render()
                        eval_episode_reward += eval_rl

                        eval_qs.append(eval_q)
                        if eval_done:
                            eval_obs = eval_env.reset()
                            eval_episode_rewards.append(eval_episode_reward)
                            eval_episode_rewards_history.append(eval_episode_reward)
                            eval_episode_reward = 0.
                #print(episode_rewards_history) 
            if (t)%7500 == 0:
                fname="/home/vaisakhs_shaj/Desktop/BIG-DATA/memoryNorm"+str(memory.nb_entries)+".pickle"
                pickle.dump(memory,open(fname,"wb"),protocol=-1)
            if t % 5000 == 0:
                print("=======saving interim model==========")
                filename="/home/vaisakhs_shaj/Desktop/MODEL/normal/tfSteps"+str(t)+".model"
                saver.save(sess,filename)
            mpi_size = MPI.COMM_WORLD.Get_size()
            
            # Log stats.
            # XXX shouldn't call np.mean on variable length lists
            duration = time.time() - start_time
            stats = agent.get_stats()
            combined_stats = stats.copy()
            combined_stats['rollout/return'] = np.mean(epoch_episode_rewards)
            combined_stats['rollout/return_history'] = np.mean(episode_rewards_history)
            combined_stats['rollout/episode_steps2'] = np.mean(epoch_episode_steps)
            combined_stats['rollout/episode_steps3'] = np.mean(epoch_episode_steps3)
            combined_stats['rollout/actions_mean'] = np.mean(epoch_actions)
            combined_stats['rollout/Q_mean'] = np.mean(epoch_qs)
            combined_stats['train/loss_actor'] = np.mean(epoch_actor_losses)
            combined_stats['train/loss_critic'] = np.mean(epoch_critic_losses)
            combined_stats['train/param_noise_distance'] = np.mean(epoch_adaptive_distances)
            combined_stats['total/duration'] = duration
            combined_stats['total/steps_per_second'] = float(t) / float(duration)
            combined_stats['total/episodes'] = episodes
            combined_stats['rollout/episodes'] = epoch_episodes
            combined_stats['rollout/actions_std'] = np.std(epoch_actions)
            # Evaluation statistics.
            
            if eval_env is not None:
                combined_stats['eval/return'] = np.mean(eval_episode_rewards)
                combined_stats['eval/return_history'] = np.mean(eval_episode_rewards_history)
                combined_stats['eval/Q'] = np.mean(eval_qs)
                combined_stats['eval/episodes'] = len(eval_episode_rewards)
               

            def as_scalar(x):
                if isinstance(x, np.ndarray):
                    assert x.size == 1
                    return x[0]
                elif np.isscalar(x):
                    return x
                else:
                    raise ValueError('expected scalar, got %s'%x)
            combined_stats_sums = MPI.COMM_WORLD.allreduce(np.array([as_scalar(x) for x in combined_stats.values()]))
            combined_stats = {k : v / mpi_size for (k,v) in zip(combined_stats.keys(), combined_stats_sums)}

            # Total statistics.
            combined_stats['total/epochs'] = epoch + 1
            combined_stats['total/steps'] = t

            for key in sorted(combined_stats.keys()):
                logger.record_tabular(key, combined_stats[key])
            logger.dump_tabular()
            logger.info('')
            logdir = logger.get_dir()
            print(logdir)
            if rank == 0 and logdir:
                if hasattr(env, 'get_state'):
                    with open(os.path.join(logdir, 'env_state.pkl'), 'wb') as f:
                        pickle.dump(env.get_state(), f)
                if eval_env and hasattr(eval_env, 'get_state'):
                    with open(os.path.join(logdir, 'eval_env_state.pkl'), 'wb') as f:
                        pickle.dump(eval_env.get_state(), f)

# Observations
'''
Reward scaling 10 worse performance
Reward scaling 0.1 better performance
Increaing batch size better performance at the cost of computational speed
'''