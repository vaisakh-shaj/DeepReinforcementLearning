import numpy as np
import tensorflow as tf
import gym
from dynamics import NNDynamicsModel
from controllers import MPCcontroller, RandomController
from cost_functions import cheetah_cost_fn, trajectory_cost_fn
import time
import logz
import os
import copy
import matplotlib.pyplot as plt
from cheetah_env import HalfCheetahEnvNew

def sample(env, 
           controller, 
           num_paths=10, 
           horizon=1000, 
           render=False,
           verbose=False):
    """
        Write a sampler function which takes in an environment, a controller (either random or the MPC controller), 
        and returns rollouts by running on the env. 
        Each path can have elements for observations, next_observations, rewards, returns, actions, etc.
    """
    paths = []
    rewards=[]
    costs=[]
    print("num_sum_path",num_paths)
    for i in range(num_paths):
        print("path :",i)
        states=list()
        actions=list()
        next_states=list()
        states.append(env.reset())
        #print(np.array(states).shape)
        totalr=0
        totalc=0
        for j in range(horizon):
            if(j%100==0):
                print(j)
            act,c=controller.get_action(states[j])
            actions.append(act)
            obs, r, done, _ = env.step(actions[j])
            next_states.append(obs)
            if j!=horizon-1:
                states.append(next_states[j])
            totalr+=r
            totalc+=c
        #print(np.array(next_states).shape)
        #print(np.array(states).shape)
        path = {'observations': np.array(states),
                       'actions': np.array(actions),
                       'next_observations': np.array(next_states)
                       }
        paths.append(path)
        rewards.append(totalr)
        costs.append(totalc)

    return paths,rewards,costs

# Utility to compute cost a path for a given cost function
def path_cost(cost_fn, path):
    return trajectory_cost_fn(cost_fn, path['observations'], path['actions'], path['next_observations'])

def compute_normalization(data):
    """
    Write a function to take in a dataset and compute the means, and stds.
    Return 6 elements: mean of s_t, std of s_t, mean of (s_t+1 - s_t), std of (s_t+1 - s_t), mean of actions, std of actions
    """
    
    """ YOUR CODE HERE """
    mean_obs=np.mean(data['observations'],axis=0)
    std_obs=np.std(data['observations'],axis=0)
    mean_deltas=np.mean(data['delta'],axis=0)
    std_deltas=np.std(data['delta'],axis=0)
    mean_actions=np.mean(data['actions'],axis=0)
    std_actions=np.std(data['actions'],axis=0)
    return mean_obs, std_obs, mean_deltas, std_deltas, mean_actions, std_actions


def plot_comparison(env, dyn_model):
    """
    Write a function to generate plots comparing the behavior of the model predictions for each element of the state to the actual ground truth, using randomly sampled actions. 
    """
    """ YOUR CODE HERE """
    pass

def train(env, 
         cost_fn,
         logdir=None,
         render=False,
         learning_rate=1e-3,
         onpol_iters=1,
         dynamics_iters=60,
         batch_size=512,
         num_paths_random=10, 
         num_paths_onpol=1, 
         num_simulated_paths=10000,
         env_horizon=1000, 
         mpc_horizon=15,
         n_layers=2,
         size=100,
         activation=tf.nn.relu,
         output_activation=None
         ):

    """

    Arguments:

    onpol_iters                 Number of iterations of onpolicy aggregation for the loop to run. 

    dynamics_iters              Number of iterations of training for the dynamics model
    |_                          which happen per iteration of the aggregation loop.

    batch_size                  Batch size for dynamics training.

    num_paths_random            Number of paths/trajectories/rollouts generated 
    |                           by a random agent. We use these to train our 
    |_                          initial dynamics model.
    
    num_paths_onpol             Number of paths to collect at each iteration of
    |_                          aggregation, using the Model Predictive Control policy.

    num_simulated_paths         How many fictitious rollouts the MPC policy
    |                           should generate each time it is asked for an
    |_                          action.

    env_horizon                 Number of timesteps in each path.

    mpc_horizon                 The MPC policy generates actions by imagining 
    |                           fictitious rollouts, and picking the first action
    |                           of the best fictitious rollout. This argument is
    |                           how many timesteps should be in each fictitious
    |_                          rollout.

    n_layers/size/activations   Neural network architecture arguments. 

    """

    logz.configure_output_dir(logdir)

    #========================================================
    # 
    # First, we need a lot of data generated by a random
    # agent, with which we'll begin to train our dynamics
    # model.

    random_controller = RandomController(env)

    paths,rewards,costs=sample(env,random_controller,num_paths_random)
    obs = np.concatenate([path["observations"] for path in paths])
    acs = np.concatenate([path["actions"] for path in paths])
    n_obs = np.concatenate([path["next_observations"] for path in paths])
    delta = n_obs-obs
    data = {'observations': obs,
                   'actions': acs,
                   'delta': delta
                   }
    

    #========================================================
    # 
    # The random data will be used to get statistics (mean
    # and std) for the observations, actions, and deltas
    # (where deltas are o_{t+1} - o_t). These will be used
    # for normalizing inputs and denormalizing outputs
    # from the dynamics network. 
    # 
    mean_obs, std_obs, mean_deltas, std_deltas, mean_actions, std_actions = compute_normalization(data)
    normalization = dict()
    normalization['observations']=[mean_obs, std_obs]
    normalization['actions']=[mean_actions, std_actions]
    normalization['delta']=[mean_deltas, std_deltas]
    #========================================================
    # 
    # Build dynamics model and MPC controllers.
    # 
    sess = tf.Session()

    dyn_model = NNDynamicsModel(env=env, 
                                n_layers=n_layers, 
                                size=size, 
                                activation=activation, 
                                output_activation=output_activation, 
                                normalization=normalization,
                                batch_size=batch_size,
                                iterations=dynamics_iters,
                                learning_rate=learning_rate,
                                sess=sess)

    mpc_controller = MPCcontroller(env=env, 
                                   dyn_model=dyn_model, 
                                   horizon=mpc_horizon, 
                                   cost_fn=cost_fn, 
                                   num_simulated_paths=num_simulated_paths)


    #========================================================
    # 
    # Tensorflow session building.
    # 
    sess.__enter__()
    tf.global_variables_initializer().run()

    #========================================================
    # 
    # Take multiple iterations of onpolicy aggregation at each iteration refitting the dynamics model to current dataset and then taking onpolicy samples and aggregating to the dataset. 
    # Note: You don't need to use a mixing ratio in this assignment for new and old data as described in https://arxiv.org/abs/1708.02596
    # 
    print("onpol_iter",onpol_iters)
    for itr in range(onpol_iters):
        """ YOUR CODE HERE """
        print(data['observations'].shape)
        #print(data['observations'].shape)
        dyn_model.fit(data)
        
        # Generate trajectories from MPC controllers
        
        pathsM,returns,costs=sample(env,mpc_controller,num_paths_onpol)
        obs = np.concatenate([path["observations"] for path in pathsM])
        acs = np.concatenate([path["actions"] for path in pathsM])
        n_obs = np.concatenate([path["next_observations"] for path in pathsM])
        delta = n_obs-obs
        data = {'observations': np.concatenate((data['observations'],
                             obs)),'actions': np.concatenate((data['actions'],
                             acs)),'delta': np.concatenate((data['delta'],
                             delta)) }


        # LOGGING
        # Statistics for performance of MPC policy using
        # our learned dynamics model
        logz.log_tabular('Iteration', itr)
        # In terms of cost function which your MPC controller uses to plan
        logz.log_tabular('AverageCost', np.mean(costs))
        logz.log_tabular('StdCost', np.std(costs))
        logz.log_tabular('MinimumCost', np.min(costs))
        logz.log_tabular('MaximumCost', np.max(costs))
        # In terms of true environment reward of your rolled out trajectory using the MPC controller
        logz.log_tabular('AverageReturn', np.mean(returns))
        logz.log_tabular('StdReturn', np.std(returns))
        logz.log_tabular('MinimumReturn', np.min(returns))
        logz.log_tabular('MaximumReturn', np.max(returns))

        logz.dump_tabular()

def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v2')
    # Experiment meta-params
    parser.add_argument('--exp_name', type=str, default='mb_mpc')
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--render', action='store_true')
    # Training args
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--onpol_iters', '-n', type=int, default=1)
    parser.add_argument('--dyn_iters', '-nd', type=int, default=150)
    parser.add_argument('--batch_size', '-b', type=int, default=512)
    # Data collection
    parser.add_argument('--random_paths', '-r', type=int, default=10)
    parser.add_argument('--onpol_paths', '-d', type=int, default=1)
    parser.add_argument('--simulated_paths', '-sp', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=int, default=1000)
    # Neural network architecture args
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=500)
    # MPC Controller
    parser.add_argument('--mpc_horizon', '-m', type=int, default=15)
    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # Make data directory if it does not already exist
    if not(os.path.exists('data')):
        os.makedirs('data')
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    # Make env
    if args.env_name is "HalfCheetah-v2":
        env = HalfCheetahEnvNew()
        cost_fn = cheetah_cost_fn
    train(env=env, 
                 cost_fn=cost_fn,
                 logdir=logdir,
                 render=args.render,
                 learning_rate=args.learning_rate,
                 onpol_iters=args.onpol_iters,
                 dynamics_iters=args.dyn_iters,
                 batch_size=args.batch_size,
                 num_paths_random=args.random_paths, 
                 num_paths_onpol=args.onpol_paths, 
                 num_simulated_paths=args.simulated_paths,
                 env_horizon=args.ep_len, 
                 mpc_horizon=args.mpc_horizon,
                 n_layers = args.n_layers,
                 size=args.size,
                 activation=tf.nn.relu,
                 output_activation=None,
                 )

if __name__ == "__main__":
    main()
'''
[60.012291897764065, 60.15172817223024, 97.78301611599068, 68.58186171067577, 75.03376735481855, 58.331157544433836, 55.72633010034349, 59.10931839227771, 95.83430801814868, 137.84664027932328, 94.90278881577235, 124.3845383772149, 77.01335085832181, 150.53915627874645, 152.9446096505178, 175.95321524732847, 168.70426886511422, 200.40177861735066, 219.36744053424266, 202.68826410485062, 147.94205198084148, 142.9901602193305, 109.41105964846157, 97.04488988702245, 88.00784161904015, 83.1454635215327, 82.75082153767983, 71.61399618385066, 74.2877803760792, 72.45300800784288, 79.23708656429694, 76.53100719000531, 75.25806603296414, 80.83825526978444, 78.16245286534314, 80.00106555527142, 88.76273540978126, 98.28612014741525, 99.59663721222547, 216.57465415920606, 213.31586171360885, 219.38498764383476, 216.3851459813369, 221.48484044018983, 216.44408320507017, 217.60958921709405, 219.01942888088, 220.51168858524494, 218.24333284506994, 215.9477673475083, 218.56191493482277, 218.82557673349348, 217.87625390888962, 227.34565944637808, 291.0625790006954, 324.3397457856642, 485.18045024546245, 540.0481201633044, 615.5341110702421, 622.2137648977123, 637.6473120208327, 673.8438595734801, 658.2685723701024, 680.6505319261628, 684.7916536742136, 681.3826775808902, 695.9858223440891, 656.4521537947767, 677.7432502842195, 677.0432213694686, 676.0195749600063, 677.3858071060415, 669.0364863764786, 707.3913582134174, 676.3596444680074, 627.5379276891022, 660.4916674774605, 577.8401346892977, 644.7345834453322, 637.8933920809015, 762.4483639085117, 639.7957588312486, 647.4329120684741, 585.0700753173021, 673.646490482827, 529.923260212984, 611.3031906264922, 768.6908976144691, 531.7950121023944, 591.1729382746225, 626.2845090106508, 562.351238157887]
'''