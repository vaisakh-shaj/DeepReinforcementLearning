import numpy as np
from cost_functions import trajectory_cost_fn
import time

class Controller():
    def __init__(self):
        pass
    # Get the appropriate action(s) for this state(s)
    def get_action(self, state):
        pass


class RandomController(Controller):
    def __init__(self, env):
        self.ac = env.action_space
    def get_action(self, state):
        """ YOUR CODE HERE """
        """ Your code should randomly sample an action uniformly from the action space """
        return self.ac.sample(),0


class MPCcontroller(Controller):
    """ Controller built using the MPC method outlined in https://arxiv.org/abs/1708.02596 """
    def __init__(self, 
				 env, 
				 dyn_model, 
				 horizon=5, 
				 cost_fn=None, 
				 num_simulated_paths=10,
				 ):
        self.env = env
        self.dyn_model = dyn_model
        self.horizon = horizon
        self.cost_fn = cost_fn
        self.num_simulated_paths = num_simulated_paths
    
    def get_action(self, state):
        """ Note: be careful to batch your simulations through the model for speed """
    	 #sampled_acts = np.array([[self.env.action_space.sample() for j in range(self.num_simulated_paths)] for i in range(self.horizon)])
        sampled_acts = np.array([[self.env.action_space.sample() for j in range(self.num_simulated_paths)] for i in range(self.horizon)])
        #sampled_acts=np.random.randint(,size=[self.horizon,self.num_simulated_paths])
        states = [np.array([state] * self.num_simulated_paths)]
        nstates = []
        for i in range(self.horizon):
             nstates.append(self.dyn_model.predict(states[-1], sampled_acts[i, :]))
             if i < self.horizon: states.append(nstates[-1])
        costs = trajectory_cost_fn(self.cost_fn, states, sampled_acts, nstates)
        return sampled_acts[0][np.argmin(costs)],min(costs)
 

		

'''
    # Without parallel predictions
    
    def get_action(self, state):
       """ YOUR CODE HERE """
		""" Note: be careful to batch your simulations through the model for speed """
       next_observations = []
       observations = []
       actions = []
       num_act=env.action_space.n
       for i in range(args.num_simulated_paths):
          print('controller_path', i)
          actions[i]=np.random.randint(num_act, size=horizon)
          observations[i][0]=state
          observations[i]=[dyn_model.predict(observations[i][j-1],actions[j-1]) for j in range(1,horizon)]
          next_observations[i][horizon-1]=dyn_model.predict(observations[i][horizon-1])
          next_observations[i][0:horizon-1]=observations[i][1:horizon]
          cost[i]=trajectory_cost_fn(cost_fn, observations[i], actions[i], next_observations[i])

        trajectories = {'observation': np.array(observations),
                       'action': np.array(actions),
                       'next_observation': np.array(next_observations),
                       'cost'=np.array(cost)}
        m=np.argmin(cost)
        return m
'''

