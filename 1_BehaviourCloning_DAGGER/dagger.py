#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
            %run run_expert.py experts/Hopper-v1.pkl Hopper-v2 --render --num_rollouts 10

Dagger and Behaviour Clonning Implementation : @uthor : vaisakhs (vaisakhs.shaj@gmail.com)


"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import matplotlib.pyplot as plt
from keras import utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()
    

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit
        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None,:])
                #print(action.shape)
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                #print(np.array(actions).shape)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}
        
        return expert_data,args
    
def dagger(expert_data,args):
    observations = []
    actions = []
    returns = []
    aggregated_data = {'observations': expert_data['observations'],'actions': expert_data['actions'] }
    import gym
    env = gym.make(args.envname)
    max_steps = 500
    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    #construct Model
    with tf.Session():
        tf_util.initialize()
        for i in range(25):
            xTr=aggregated_data['observations']
            yTr=aggregated_data['actions']
            print("hellooooooo",yTr.shape)
            #print(xTr.shape)
            #print(yTr.shape)
            yTr=np.reshape(yTr,(yTr.shape[0],yTr.shape[2]))
            
            
            
            model = Sequential()
            model.add(Dense(120, input_dim=xTr.shape[1], init="uniform",
                activation="relu"))
            model.add(Dropout(0.5))
            model.add(Dense(120, input_dim=xTr.shape[1], init="uniform",
                activation="relu"))
            model.add(Dropout(0.5))
            model.add(Dense(120, init="uniform", activation="relu"))
            model.add(Dropout(0.5))
            model.add(Dense(yTr.shape[1]))
            
            #compile Model
            model.compile(loss='msle',
                          optimizer='adam',
                          metrics=['accuracy'])
            model.save_weights(r"C:\Users\vaisakhs\Desktop\GITHUB\DeepReinforcementLearning\BehaviourCloning-DAGGER\Policies\policyDNN.h5")
            
            history=model.fit(xTr, yTr,
                      epochs=10,
                      batch_size=1000,validation_split=0.1)
            #print(history.history['loss'])
            
            
            
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0
            steps = 0
            while not done:
                #print("hello")
                true_action = policy_fn(obs[None,:])
                predicted_action = model.predict(np.array(obs[None,:]))
                observations.append(obs)
                actions.append(true_action)
                #print(true_action.shape)
                obs, r, done, _ = env.step(predicted_action)
                #print(obs)
                #done=False
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)
    
            print('returns', returns)
            print('mean return', np.mean(returns))
            print('std of return', np.std(returns))
    
            aggregated_data = {'observations': np.concatenate((aggregated_data['observations'],
                             np.array(observations))),'actions': np.concatenate((aggregated_data['actions'],
                             np.array(actions))) }
        
    return aggregated_data

    
    

if __name__ == '__main__':
    expert_data,args=main()
    ed2=dagger(expert_data,args)
