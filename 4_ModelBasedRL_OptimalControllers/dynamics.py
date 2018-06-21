import tensorflow as tf
import numpy as np

# Predefined function to build a feedforward neural network
def build_mlp(input_placeholder, 
              output_size,
              scope, 
              n_layers=2, 
              size=500, 
              activation=tf.tanh,
              output_activation=None
              ):
    out = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            out = tf.layers.dense(out, size, activation=activation)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out

def normalize(data, mean, std):
    return (data - mean) / (std + 1e-10)

def denormalize(data, mean, std):
    return data * (std + 1e-10) + mean 

class NNDynamicsModel():
    def __init__(self, 
                 env, 
                 n_layers,
                 size, 
                 activation, 
                 output_activation, 
                 normalization,
                 batch_size,
                 iterations,
                 learning_rate,
                 sess
                 ):
        
        """ Note: Be careful about normalization """
        # https://stackoverflow.com/questions/37770911/tensorflow-creating-a-graph-in-a-class-and-running-it-ouside
        self.normalization=normalization
        self.iterations=iterations
        self.batch_size=batch_size
        self.sess=sess
        ob_dim = env.observation_space.shape[0] #local variables of init just for convinience
        ac_dim = env.action_space.shape[0]
        self.sy_ob = tf.placeholder(shape=[None, ob_dim], name="ob", dtype=tf.float32)
        self.sy_ac = tf.placeholder(shape=[None, ac_dim], name="ac", dtype=tf.float32)
        self.delta = tf.placeholder(shape=[None, ob_dim], name="del", dtype=tf.float32)
        ob_ac = tf.concat([self.sy_ob,self.sy_ac],axis=1)
        self.delta_prediction=build_mlp(ob_ac ,ob_dim,'trans_dyna',
                                   n_layers,size,activation,output_activation)
        loss=tf.losses.mean_squared_error(labels=self.delta,predictions=self.delta_prediction)
        self.dyna_update_op=tf.train.AdamOptimizer(learning_rate).minimize(loss)

    def fit(self, data):
        """
        Write a function to take in a dataset of (unnormalized)states, (unnormalized)actions, (unnormalized)next_states and 
        fit the dynamics model going from normalized states, normalized actions to normalized state differences (s_t+1 - s_t)
        """
        #paths=data
        obs = data['observations']
        delta = data["delta"]
        acs = data["actions"]
        
        
        ### normalize
        obs = normalize(obs,self.normalization['observations'][0],self.normalization['observations'][1])
        delta = normalize(delta,self.normalization['delta'][0],self.normalization['delta'][1])
        acs = normalize(acs,self.normalization['actions'][0],self.normalization['actions'][1])
        
        
        train_count = len(obs)  
        N_EPOCHS=50
        for i in range(1, N_EPOCHS + 1):       # tf.data /tf.train.batch /tf.train.shuffle_batch -- later
            print("epoch: ",i)
            done=False
            start=0;end=0
            while(not done):
                start=end
                end=min(start+self.batch_size,train_count)
                #print(start+self.batch_size,train_count)
                #print(end)
                if(end==train_count):
                    done=True
                self.sess.run(self.dyna_update_op, feed_dict={self.sy_ob:obs[start:end], self.sy_ac:acs[start:end], self.delta:delta[start:end] })
                

    def predict(self, states, actions):
        """ Write a function to take in a batch of (unnormalized) states and (unnormalized) actions 
        and return the (unnormalized) next states as predicted by using the model """
        obs = normalize(states,self.normalization['observations'][0],self.normalization['observations'][1])
        #delta = normalize(delta,normalization['delta'])
        acs = normalize(actions,self.normalization['actions'][0],self.normalization['actions'][1])
        done=False
        start=0;end=0
        test_count=len(states)
        #print(test_count)
        prediction=self.sess.run(self.delta_prediction, feed_dict={self.sy_ob:obs, self.sy_ac:acs })
            
        
        return denormalize(prediction,self.normalization['delta'][0],self.normalization['delta'][1]) + states  
