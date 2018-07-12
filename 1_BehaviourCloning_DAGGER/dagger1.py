import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import matplotlib.pyplot as plt
from keras.layers.advanced_activations import LeakyReLU, PReLU
import os


def get_data(args, init_observations=None,render=True):
    # if init_observations is None ---> generates expert data
    # if init_observations are fed ---> returns expert actions
    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')
    if init_observations is not None:
        print('initial observations: ', init_observations.shape)
    else:
        print('No initial observations: ')
    with tf.Session():
        tf_util.initialize()
        import gym
        env = gym.make(args.envname)
        obs = env.reset()
        max_steps = args.max_timesteps or env.spec.timestep_limit
        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            print('iter', i)
            done = False
            totalr = 0.
            steps = 0
            while not done:
                if init_observations is not None:
                    obs = init_observations[steps]
                action = policy_fn(np.array(obs[None, :]))
                # print(action.shape)
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                # print(r)
                totalr += r
                steps += 1
                if render:
                    env.render()
                if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
                if init_observations is not None:
                    done = False
                    if steps == len(init_observations):
                        break
                else:
                    if steps >= max_steps:
                        break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}

        return expert_data

def generate_model_samples(model, args, render=False):
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
            action = model.predict(obs[None, :])
            # print(action.shape)
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            # print(r)
            totalr += r
            steps += 1
            if render:
                env.render()
            if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

    expert_data = {'observations': np.array(observations),
                   'actions': np.array(actions)}

    return expert_data, returns

def DNN_model(input_size, output_size):
    from keras import utils
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation
    from keras.layers.advanced_activations import LeakyReLU, PReLU
    # construct Model
    model = Sequential()
    model.add(Dense(100, input_dim=input_size, init="uniform",
                    activation="linear"))
    model.add(LeakyReLU(alpha=.01))
    model.add(Dropout(0.2))
    model.add(Dense(80, init="uniform", activation="linear"))
    model.add(LeakyReLU(alpha=.01))
    model.add(Dropout(0.2))
    model.add(Dense(40, init="uniform", activation="linear"))
    model.add(LeakyReLU(alpha=.01))
    model.add(Dropout(0.2))
    model.add(Dense(output_size))
    # compile Model
    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

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

    expert_data = get_data(args)
    x = expert_data['observations']
    y = expert_data['actions']
    input_size = x.shape[1]
    output_size = y.shape[2]
    print(input_size, output_size)
    y = np.reshape(y, (y.shape[0], y.shape[2]))
    obs = x
    actions = y
    model = DNN_model(input_size, output_size)

    for i in range(10):
        if (os.path.isfile('./Policies/policyDagger.h5')):
            model.load_weights('./Policies/policyDagger.h5')
            print('loaded previous weights')
        print('fitting the model..')
        print('iteration ', i)
        model.fit(obs, actions, epochs=i+10, batch_size=256, validation_split=0.1)
        model.save_weights("./Policies/policyDagger.h5")
        print('weights saved')
        print('generating samples from the model..')
        model_samples, _ = generate_model_samples(model, args, render=False)
        obs_samples = model_samples['observations']
        print('getting expert actions..')
        expert_data = get_data(args, init_observations=obs_samples, render=False)
        expert_actions = expert_data['actions']
        expert_actions = np.reshape(expert_actions, (expert_actions.shape[0], expert_actions.shape[2]))
        print(expert_actions.shape)
        print(actions.shape)
        print(obs.shape)
        print(obs_samples.shape)
        obs = np.concatenate((obs, obs_samples))
        actions = np.concatenate((actions, expert_actions))

    model.load_weights('./Policies/policyDagger.h5')
    print('running the final model..')
    _, _ = generate_model_samples(model, args, render=True)


if __name__ == '__main__':
    main()







