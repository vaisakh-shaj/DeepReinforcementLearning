#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
            python ss

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import matplotlib.pyplot as plt


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
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
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
        
        return expert_data
    
def behaviourCloning(expert_data):
    from keras import utils
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation
    xTr=expert_data['observations']
    yTr=expert_data['actions']

    import numpy as np
    #print(xTr.shape)
    #print(yTr.shape)
    yTr=np.reshape(yTr,(yTr.shape[0],yTr.shape[2]))
    #construct Model
    model = Sequential()
    model.add(Dense(200, input_dim=376, init="uniform",
        activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(200, init="uniform", activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(17))
    model.add(Activation("softmax"))
    #compile Model
    model.compile(loss='mean_squared_error',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    
    history=model.fit(xTr, yTr,
              epochs=10,
              batch_size=256)
    
    
    ######### Training Error Plot
    det=list()
    for i,e in enumerate(history.history['loss'],0):
        det.append(e)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(range(len(det)),det, linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=12)
    for x, y in zip( range(len(det)), det):
        plt.text(x, y, str(y), color="blue", fontsize=12)
    plt.title("Loss Vs Epoch Curve")
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.show()

    
    

if __name__ == '__main__':
    expert_data=main()
    behaviourCloning(expert_data)
