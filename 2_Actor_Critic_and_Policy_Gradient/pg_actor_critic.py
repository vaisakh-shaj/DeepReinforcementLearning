import numpy as np
import tensorflow as tf
import gym
import logz
import scipy.signal
import os
import time
import inspect
from multiprocessing import Process


# ============================================================================================#
# Utilities
# ============================================================================================#
def normalize(data, mean=0.0, std=1.0):
    n_data = (data - np.mean(data)) / (np.std(data) + 1e-8)
    return n_data * (std + 1e-8) + mean


# ========================= Critic network =================================#
# input is a state(observation)
# output is a value of that state (Expected Average return from given state
# undera given policy (Actor in this case))
# ==========================================================================#
def critic(input_placeholder):
    z = input_placeholder
    layers = 3
    sizes = [20, 15, 10]
    for i in range(layers+1):
        with tf.variable_scope('dense', reuse=tf.AUTO_REUSE):
            if i < layers:
                with tf.variable_scope('mlp' + str(i), reuse=tf.AUTO_REUSE):
                    z = tf.layers.dense(z, units=sizes[i],
                                        activation=None)  # weight matrix automatically created by the model
                    z = tf.layers.dropout(z, rate=0.00)
            else:
                with tf.variable_scope('mlp' + str(i), reuse=tf.AUTO_REUSE):
                    value = tf.layers.dense(z, units=1, name='state_value')
    return value


# ============================= Actor network ==================================#
# input is a state(observation)
# output returns logits over actions in descrete case
# mean in case of continuous action space
# ==============================================================================#
def build_mlp(
        input_placeholder,
        output_size,
        scope,
        n_layers=2,
        size=64,
        activation=tf.tanh,
        output_activation=None,
        training=False
):
    # ========================================================================================#
    #                           ----------SECTION 3----------
    # Network building
    #
    # Your code should make a feedforward neural network (also called a multilayer perceptron)
    # with 'n_layers' hidden layers of size 'size' units.
    #
    # The output layer should have size 'output_size' and activation 'output_activation'.
    #
    # Hint: use tf.layers.dense
    # ========================================================================================#
    z = input_placeholder
    sizes = [20, 10]
    n_layers = 2
    for i in range(0, n_layers + 1):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            if i < n_layers:
                with tf.variable_scope('mlp' + str(i), reuse=tf.AUTO_REUSE):
                    z = tf.layers.dense(z, units=sizes[i],
                                        activation=activation)  # weight matrix automatically created by the model
                    z = tf.layers.dropout(z, rate=0.00)  # Boolean variable training can
                    # be set to false to avoid this step during inference
            else:
                with tf.variable_scope('mlp' + str(n_layers), reuse=tf.AUTO_REUSE):
                    logits = tf.layers.dense(z, units=output_size, name='logits')
                    y = tf.nn.softmax(logits, name='ybar')
    return logits


def pathlength(path):
    return len(path["reward"])


# ============================================================================================#
# Policy Gradient
# ============================================================================================#
def train_PG(exp_name='',
             env_name='CartPole-v0',
             n_iter=100,
             gamma=1.0,
             test=False,
             min_timesteps_per_batch=1000,
             max_path_length=None,
             learning_rate=5e-3,
             reward_to_go=True,
             animate=True,
             logdir=None,
             seed=0,
             # network arguments
             n_layers=1,
             size=32
             ):
    start = time.time()

    # Configure output directory for logging
    logz.configure_output_dir(logdir)

    # Log experimental parameters
    args = inspect.getargspec(train_PG)[0]
    locals_ = locals()
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_params(params)

    # Set random seeds
    tf.set_random_seed(seed)
    np.random.seed(seed)

    # Make the gym environment
    env = gym.make(env_name)

    # Is this env continuous, or discrete?
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # Maximum length for episodes
    max_path_length = max_path_length or env.spec.max_episode_steps

    # ========================================================================================#
    # Notes on notation:
    #
    # Symbolic variables have the prefix sy_, to distinguish them from the numerical values
    # that are computed later in the function
    #
    # Prefixes and suffixes:
    # ob - observation
    # ac - action
    # _no - this tensor should have shape (batch size /n/, observation dim)
    # _na - this tensor should have shape (batch size /n/, action dim)
    # _n  - this tensor should have shape (batch size /n/)
    #
    # Note: batch size /n/ is defined at runtime, and until then, the shape for that axis
    # is None
    # ========================================================================================#

    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]
    print('observation dim: ', ob_dim)
    print('action dim: ', ac_dim)
    print('action space: ', discrete)
    # print("hellooooooo",ac_dim,env.action_space.shape)
    # ========================================================================================#
    #                           ----------SECTION 4----------
    # Placeholders
    #
    # Need these for batch observations / actions / advantages in policy gradient loss function.
    # ========================================================================================#

    sy_ob_no = tf.placeholder(shape=[None, ob_dim], name="ob", dtype=tf.float32)
    if discrete:
        sy_ac_na = tf.placeholder(shape=[None, ac_dim], name="ac", dtype=tf.int32)
    else:
        sy_ac_na = tf.placeholder(shape=[None, ac_dim], name="ac", dtype=tf.float32)

        # Define a placeholder for advantages
    sy_adv_n = tf.placeholder(dtype=tf.float32, shape=[None], name="adv")

    # ========================================================================================#
    #                           ----------SECTION 4----------
    # Networks
    #
    # Make symbolic operations for
    #   1. Policy network outputs which describe the policy distribution.
    #       a. For the discrete case, just logits for each action.
    #
    #       b. For the continuous case, the mean / log std of a Gaussian distribution over
    #          actions.
    #
    #      Hint: use the 'build_mlp' function you defined in utilities.
    #
    #      Note: these ops should be functions of the placeholder 'sy_ob_no'
    #
    #   2. Producing samples stochastically from the policy distribution.
    #       a. For the discrete case, an op that takes in logits and produces actions.
    #
    #          Should have shape [None]
    #
    #       b. For the continuous case, use the reparameterization trick:
    #          The output from a Gaussian distribution with mean 'mu' and std 'sigma' is
    #
    #               mu + sigma * z,         z ~ N(0, I)
    #
    #          This reduces the problem to just sampling z. (Hint: use tf.random_normal!)
    #
    #          Should have shape [None, ac_dim]
    #
    #      Note: these ops should be functions of the policy network output ops.
    #
    #   3. Computing the log probability of a set of actions that were actually taken,
    #      according to the policy.
    #
    #      Note: these ops should be functions of the placeholder 'sy_ac_na', and the
    #      policy network output ops.
    #
    # ========================================================================================#

    if discrete:
        # YOUR_CODE_HERE
        sy_logits_na = build_mlp(sy_ob_no, ac_dim, scope="build_nn", n_layers=n_layers,
                                 size=size,
                                 activation=tf.nn.relu)
        sy_sampled_ac = tf.one_hot(tf.squeeze(tf.multinomial(sy_logits_na, 1)),
                                   ac_dim)  # Hint: Use the tf.multinomial op
        # batch_size x ac_dim

        sy_logprob_n = tf.nn.softmax_cross_entropy_with_logits_v2(labels=sy_ac_na, logits=sy_logits_na)
        # batch_size ---> log probability for each action

        # Learned from https://github.com/InnerPeace-Wu/
        # # Another way to do it
        # N = tf.shape(sy_ob_no)[0]
        # sy_prob_na = tf.nn.softmax(sy_logits_na)
        # sy_logprob_n = tf.log(tf.gather_nd(sy_prob_na, tf.stack((tf.range(N), sy_ac_na), axis=1)))
    else:
        # YOUR_CODE_HERE
        sy_mean = build_mlp(sy_ob_no, ac_dim, scope="build_nn", n_layers=n_layers,
                            size=size,
                            activation=tf.nn.relu)
        sy_logstd = tf.Variable(tf.zeros(ac_dim), name='logstd',
                                dtype=tf.float32)
        sy_std = tf.exp(sy_logstd)
        sy_sampled_ac = sy_mean + tf.multiply(sy_std, tf.random_normal(tf.shape(sy_mean)))
        sy_z = (sy_ac_na - sy_mean) / sy_std

        sy_logprob_n = 0.5 * tf.reduce_sum(tf.square(sy_z), axis=1)
        # sy_logprob_n = 0.5*tf.reduce_sum(tf.squared_difference(tf.div(sy_mean,sy_std),
        # tf.div(sy_ac_na,sy_std)))  # Hint: Use the log probability under a multivariate gaussian.

    # ========================================================================================#
    #                           ----------SECTION 4----------
    # Loss Function and Training Operation
    # ========================================================================================#

    # loss = tf.reduce_sum(tf.multiply(tf.nn.softmax_cross_entropy_with_logits_v2(labels=sy_ac_na,logits=sy_logits_na),sy_adv_n))
    # Loss function that we'll differentiate to get the policy gradient.

    loss = tf.reduce_sum(tf.multiply(sy_logprob_n, sy_adv_n))
    actor_update_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    actor_params = tf.trainable_variables()

    # ========================================================================================#
    # critic graph
    # Loss and training operations
    # ========================================================================================#

    predict_value = critic(sy_ob_no)
    sy_target_value = tf.placeholder(dtype=tf.float32, shape=[None], name="target_value")
    predict_value = tf.squeeze(predict_value)
    rms_loss = tf.reduce_mean(tf.squared_difference(predict_value, sy_target_value))
    critic_update_op = tf.train.AdamOptimizer(learning_rate).minimize(rms_loss)
    critic_params = tf.trainable_variables()[len(actor_params):]

    # ========================================================================================#
    # Tensorflow Engineering: Config, Session, Variable initialization
    # ========================================================================================#

    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)

    sess = tf.Session(config=tf_config)
    sess.__enter__()  # equivalent to `with sess:`

    actor_saver = tf.train.Saver(actor_params, max_to_keep=1)
    critic_saver = tf.train.Saver(critic_params, max_to_keep=1)

    checkpoint_actor_dir = os.path.join(os.curdir, 'Actor_GAE_0.7' + str(env_name))
    if not os.path.exists(checkpoint_actor_dir):
        os.makedirs(checkpoint_actor_dir)
    actor_prefix = os.path.join(checkpoint_actor_dir, "model.ckpt")
    ckpt_1 = tf.train.get_checkpoint_state(checkpoint_actor_dir)

    checkpoint_critic_dir = os.path.join(os.curdir, 'Critic_GAE_0.7' + str(env_name))
    if not os.path.exists(checkpoint_critic_dir):
        os.makedirs(checkpoint_critic_dir)
    critic_prefix = os.path.join(checkpoint_critic_dir, "model.ckpt")
    ckpt_2 = tf.train.get_checkpoint_state(checkpoint_critic_dir)

    if ckpt_1 and tf.train.checkpoint_exists(ckpt_1.model_checkpoint_path):
        print("Reading actor parameters from %s" % ckpt_1.model_checkpoint_path)
        actor_saver.restore(sess, ckpt_1.model_checkpoint_path)

    if ckpt_2 and tf.train.checkpoint_exists(ckpt_2.model_checkpoint_path):
        print("Reading critic parameters from %s" % ckpt_2.model_checkpoint_path)
        critic_saver.restore(sess, ckpt_2.model_checkpoint_path)

    uninitialized_vars = []
    for var in tf.global_variables():
        try:
            sess.run(var)
        except tf.errors.FailedPreconditionError:
            uninitialized_vars.append(var)

    if len(uninitialized_vars) > 0:
        init_new_vars_op = tf.variables_initializer(uninitialized_vars)
        sess.run(init_new_vars_op)

    def testing():
        print('testing..')
        ob = env.reset()
        steps = 0
        total_r = 0
        while True:
            one_hot_ac = sess.run(sy_sampled_ac, feed_dict={sy_ob_no: ob[None]})
            if discrete:
                ac = int(np.argmax(one_hot_ac))
            else:
                ac = one_hot_ac
            ob, rew, done, _ = env.step(ac)
            env.render()
            total_r += rew
            steps += 1
            if steps > max_path_length:
                break
        print(steps, total_r)
        return steps, total_r

    # ========================================================================================#
    # Training Loop
    # ========================================================================================#

    if test:
        testing()
        return

    total_timesteps = 0

    best_steps, best_rew = testing()
    # best_rew = 0

    for itr in range(n_iter):
        print("********** Iteration %i ************" % itr)
        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            ob = env.reset()
            obs, acs, rewards = [], [], []
            next_obs = []
            animate_this_episode = (len(paths) == 0 and (itr % 30 == 0) and animate)
            steps = 0
            while True:
                if animate_this_episode:
                    env.render()
                    time.sleep(0.05)
                obs.append(ob)
                one_hot_ac = sess.run(sy_sampled_ac, feed_dict={sy_ob_no: ob[None]})

                if discrete:
                    ac = int(np.argmax(one_hot_ac))
                else:
                    ac = one_hot_ac
                    # print("helloooo",ac)
                acs.append(one_hot_ac)
                next_ob, rew, done, _ = env.step(ac)  # transition dynamics P(s_t+1/s_t,a_t), r(s_t+1/s_t,a_t)
                next_obs.append(next_ob)
                ob = next_ob
                rewards.append(rew)
                steps += 1
                if done or steps > max_path_length:
                    break
            path = {"observation": np.array(obs),
                    "reward": np.array(rewards),
                    "action": np.array(acs),
                    "next_observation":np.array(next_obs)}
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > min_timesteps_per_batch:
                break
        total_timesteps += timesteps_this_batch

        # Build arrays for observation, action for the policy gradient update by concatenating
        # across paths
        ob_no = np.concatenate([path["observation"] for path in paths])
        next_ob_no = np.concatenate([path["next_observation"] for path in paths])
        rew_no = np.concatenate([path["reward"] for path in paths])
        ac_na = np.concatenate([path["action"] for path in paths])
        ac_na = ac_na.reshape([-1, ac_dim])
        print("helloooo", ac_na.shape)

        # ======================== Finding target values ===================================#
        # target = r(s,a) + gamma* V(s') - V(s)
        # This estimate has less variance but is biased. Alternatively
        # we can go for n-step returns or GAE(Generalised Advantage Estimation)
        # ==================================================================================#

        next_values = sess.run(predict_value, feed_dict={sy_ob_no: next_ob_no})
        target_values = rew_no + gamma*next_values

        # fit critic with target r(s,a) + gamma*V(s')
        print('updating the critic params..')
        sess.run(critic_update_op, feed_dict={sy_ob_no:ob_no, sy_target_value:target_values})

        current_values = sess.run(predict_value, feed_dict={sy_ob_no:ob_no})
        next_values = sess.run(predict_value, feed_dict={sy_ob_no: next_ob_no})
        adv_n = rew_no + gamma*next_values - current_values


        # ====================== Generalized Advatage Estimation =========================== #

        # A(s_t, a_t) = sum_{t'=t}^{t'=inf} (gamma*lambda)^{t'-t} delta_{t'}, where
        # delta_{t} = r(s_t, a_t) + gamma*V(s_{t+1}) - V(s_t)
        # ================================================================================== #

        q_n = list()
        GAE = True

        if GAE:
            ind = 0
            lam = 0.7
            for path in paths:
                pLen = pathlength(path)
                q_p = np.zeros(pLen)
                q_p[pLen - 1] = adv_n[ind + pLen - 1]
                for t in reversed(range(pLen - 1)):
                    q_p[t] = adv_n[ind + t] + (gamma*lam) * q_p[t + 1]
                q_p = np.array(q_p)
                q_n.append(q_p)
                ind += pLen

        # =========================== n-step returns =========================================#
        # Consider only the n-step returns instead of until the end of episode.
        # Variance reduction technique
        # adv(s_t) = sum_{t'=t}^(t+n) gamma^{t'-t}*r(t') + gamma^{n} V(s_{t+n}) - V(s_t)
        # ====================================================================================#

        n_step_returns = False

        if n_step_returns:
            n = 100
            value_paths = []
            for path in paths:
                ob = path['observation']
                pLen = pathlength(path)
                values = sess.run(predict_value, feed_dict={sy_ob_no:ob})
                x = {}
                x['value'] = values
                value_paths.append(x)

            for ind, path in enumerate(paths):
                pLen = pathlength(path)
                q_p = np.zeros(pLen)
                rew = path['reward']
                values = value_paths[ind]['value']
                for i in range(pLen):
                    start = i
                    end = min(start+n, pLen-1)
                    for j, r in enumerate(rew[start:end]):
                        q_p[i] += pow(gamma, j)*r
                q_p[i] += pow(gamma, n) * values[end]
                q_p[i] -= values[start]
                q_p = np.array(q_p)
                q_n.append(q_p)

        q_n = np.concatenate(q_n)
        adv_n = q_n.copy()

        # ====================================================================================#
        #                           ----------SECTION 4----------
        # Performing the Policy Update
        # ====================================================================================#

        # Call the update operation necessary to perform the policy gradient update based on
        # the current batch of rollouts.
        #
        # For debug purposes, you may wish to save the value of the loss function before
        # and after an update, and then log them below.

        # Log diagnostics
        returns = [path["reward"].sum() for path in paths]
        ep_lengths = [pathlength(path) for path in paths]

        if np.mean(returns) > best_rew:
            best_rew = np.mean(returns)
            print('saving actor to ', actor_prefix)
            actor_saver.save(sess, actor_prefix)
            print('saving critic to ', critic_prefix)
            critic_saver.save(sess, critic_prefix)

        sess.run(actor_update_op, feed_dict={sy_ac_na: ac_na, sy_ob_no: ob_no, sy_adv_n: adv_n})

        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("Iteration", itr)
        logz.log_tabular("AverageReturn", np.mean(returns))
        logz.log_tabular("StdReturn", np.std(returns))
        logz.log_tabular("MaxReturn", np.max(returns))
        logz.log_tabular("MinReturn", np.min(returns))
        logz.log_tabular("EpLenMean", np.mean(ep_lengths))
        logz.log_tabular("EpLenStd", np.std(ep_lengths))
        logz.log_tabular("TimestepsThisBatch", timesteps_this_batch)
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        logz.dump_tabular()
        logz.pickle_tf_vars()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v2')
    parser.add_argument('--exp_name', type=str, default='GAE_lambda_0.7')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=0.999)
    parser.add_argument('--n_iter', '-n', type=int, default=10000)
    parser.add_argument('--batch_size', '-b', type=int, default=5000)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=32)
    parser.add_argument('--test', '-t', action='store_true', default=False)
    args = parser.parse_args()

    if not (os.path.exists('data_actor_critic_GAE')):
        os.makedirs('data_actor_critic_GAE')
    logdir = args.exp_name + '_' + args.env_name + '_' + str(args.test)
    logdir = os.path.join('data_actor_critic_GAE', logdir)
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    max_path_length = args.ep_len if args.ep_len > 0 else None
    train_PG(
        exp_name=args.exp_name,
        env_name=args.env_name,
        n_iter=args.n_iter,
        gamma=args.discount,
        test=args.test,
        min_timesteps_per_batch=args.batch_size,
        max_path_length=max_path_length,
        learning_rate=args.learning_rate,
        reward_to_go=args.reward_to_go,
        animate=args.render,
        logdir=os.path.join(logdir, '%d' % 0),
        seed=0,
        n_layers=args.n_layers,
        size=args.size
    )

if __name__ == "__main__":
    main()