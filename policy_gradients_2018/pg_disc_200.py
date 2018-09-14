# Discrete Policy gradient model
# Action range (-1, 1) divided into 200 levels
# step_size = 1/100 = 0.01, disc_levels = 100
# change the start and stop points to change the range to (0,1)
# each nueron in the penultimate layer has a softmax layer that categorizes
# each action value into one of the 200 bins
# Total error is the sum of log_loss errors across all softmax layers in the output.
# dict_to_list_full --> to get all the 358 observations in the dictionary
# dict_to_list ---> to get the default 158 actions given in the competition
# supervised_train = True, for supervised training
# supervised_log_loss_op --> operation to train the model if we have experimental data
# test = True, to verify the cumulative reward
# exp = True, to generate the data from the existing model
# (test, exp)= (False, False) to train the model with the given config params
# You can use the reward scaling by multiplying the reward by some factor, while training
# use submit=True to submit the result to crowdAI
# model is stored in PG_DISCRETE_200_GLOBAL folder



import gym
from osim.env import ProstheticsEnv
import numpy as np
import tensorflow as tf
import gym
import logz
import os
import time
import inspect

def dict_to_list_full(state_desc):
    res = []

    # Body Observations
    for info_type in ['body_pos', 'body_pos_rot',
                      'body_vel', 'body_vel_rot',
                      'body_acc', 'body_acc_rot']:
        for body_part in ['calcn_l', 'talus_l', 'tibia_l', 'toes_l',
                          'femur_l', 'femur_r', 'head', 'pelvis',
                          'torso', 'pros_foot_r', 'pros_tibia_r']:
            res += state_desc[info_type][body_part]

    # Joint Observations
    # Neglecting `back_0`, `mtp_l`, `subtalar_l` since they do not move
    for info_type in ['joint_pos', 'joint_vel', 'joint_acc']:
        for joint in ['ankle_l', 'ankle_r', 'back', 'ground_pelvis',
                      'hip_l', 'hip_r', 'knee_l', 'knee_r']:
            res += state_desc[info_type][joint]

    # Muscle Observations
    for muscle in ['abd_l', 'abd_r', 'add_l', 'add_r',
                   'bifemsh_l', 'bifemsh_r', 'gastroc_l',
                   'glut_max_l', 'glut_max_r',
                   'hamstrings_l', 'hamstrings_r',
                   'iliopsoas_l', 'iliopsoas_r', 'rect_fem_l', 'rect_fem_r',
                   'soleus_l', 'tib_ant_l', 'vasti_l', 'vasti_r']:
        res.append(state_desc['muscles'][muscle]['activation'])
        res.append(state_desc['muscles'][muscle]['fiber_force'])
        res.append(state_desc['muscles'][muscle]['fiber_length'])
        res.append(state_desc['muscles'][muscle]['fiber_velocity'])

    # Force Observations
    # Neglecting forces corresponding to muscles as they are redundant with
    # `fiber_forces` in muscles dictionaries
    for force in ['AnkleLimit_l', 'AnkleLimit_r',
                  'HipAddLimit_l', 'HipAddLimit_r',
                  'HipLimit_l', 'HipLimit_r', 'KneeLimit_l', 'KneeLimit_r']:
        res += state_desc['forces'][force]

        # Center of Mass Observations
        res += state_desc['misc']['mass_center_pos']
        res += state_desc['misc']['mass_center_vel']
        res += state_desc['misc']['mass_center_acc']

    return res

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

# ============================================================================================#
# Utilities
# ============================================================================================#
def normalize(data, mean=0.0, std=1.0):
    n_data = (data - np.mean(data)) / (np.std(data) + 1e-8)
    return n_data * (std + 1e-8) + mean


def build_mlp(
        action_bound,
        input_placeholder,
        output_size,
        scope,
        n_layers=2,
        size=64,
        activation=tf.tanh,
        disc_levels = 50,
        batch_size=1
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

    sizes = [150, 100, 50]
    z = input_placeholder
    for i in range(1, n_layers + 1):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            if i != n_layers:
                with tf.variable_scope('mlp' + str(i), reuse=tf.AUTO_REUSE):
                    z = tf.layers.dense(z, units=sizes[i-1],
                                        activation=activation)  # weight matrix automatically created by the model
                    z = tf.layers.dropout(z, rate=0.1)  # Boolean variable training can
                    # be set to false to avoid this step during inference
            else:
                with tf.variable_scope('mlp' + str(n_layers), reuse=tf.AUTO_REUSE):
                    logits = tf.layers.dense(z, units=output_size, name='logits', activation=None)

    final_logits = []
    batch_size = tf.shape(z)[0]
    for i in range(output_size):
        with tf.variable_scope('softmax_'+str(i), reuse=tf.AUTO_REUSE):
            ind_y = tf.fill(dims=[batch_size], value=i)
            ind_x = tf.range(batch_size)
            ind = tf.stack([ind_x, ind_y], axis=1)
            logits_i = tf.gather_nd(logits, ind)
            logits_i = tf.expand_dims(logits_i, axis=1) # batch_size X 1
            dense_i = tf.layers.dense(logits_i, disc_levels, name='logits_'+str(i), activation=None) # batch_size X disc_levels
            final_logits.append(dense_i)

    logits = tf.stack(final_logits, axis=1) # batch_size X ac_dim X disc_levels
    # print(logits.get_shape())
    return logits


def pathlength(path):
    return len(path["reward"])


def train_PG(exp_name='',
             env_name='CartPole-v0',
             n_iter=100,
             gamma=1.0,
             min_timesteps_per_batch=1000,
             max_path_length=None,
             learning_rate=5e-3,
             reward_to_go=True,
             animate=True,
             logdir=None,
             normalize_advantages=True,
             nn_baseline=False,
             seed=0,
             # network arguments
             n_layers=1,
             size=32,
             test=False,
             disc_levels = 25
             ):
    start = time.time()

    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
    sess = tf.Session(config=tf_config)
    sess.__enter__()  # equivalent to `with sess:`
    # pylint: disable=E1101

    logz.configure_output_dir(logdir)
    # Log experimental parameters
    args = inspect.getargspec(train_PG)[0]
    locals_ = locals()
    params = {k: locals_[k] if k in locals_ else None for k in args}
    params['env_name'] = 'Prosthetic_3D'
    print('params: ', params)
    logz.save_params(params)

    args = inspect.getargspec(train_PG)[0]

    # Set random seeds
    tf.set_random_seed(seed)
    np.random.seed(seed)

    # Make the gym environment
    env = env_name

    # Maximum length for episodes
    max_path_length = max_path_length or env.spec.timestep_limit

    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]
    print('observation dim: ', ob_dim)
    print('action dim: ', ac_dim)

    # ========================================================================================#
    #                           ----------SECTION 4----------
    # Placeholders
    #
    # Need these for batch observations / actions / advantages in policy gradient loss function.
    # ========================================================================================#

    sy_ob_no = tf.placeholder(shape=[None, ob_dim], name="ob", dtype=tf.float32)
    sy_ac_na = tf.placeholder(shape=[None, ac_dim], name="ac", dtype=tf.int32)

    # Define a placeholder for advantages
    sy_adv_n = tf.placeholder(dtype=tf.float32, shape=[None], name="adv")
    sy_logits_na = build_mlp(env.action_space.high, sy_ob_no, ac_dim, scope="build_nn", n_layers=n_layers,
                             size=size,
                             activation=None,
                             disc_levels = 2*disc_levels)   # batch_size X ac_dim X disc_levels

    labels = tf.one_hot(sy_ac_na, depth=2*disc_levels, axis=-1)   # batch_size X ac_dim X disc_levels
    losses = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=sy_logits_na, labels=labels), axis=1)
    probs = tf.nn.softmax(sy_logits_na)
    ac_indices = tf.argmax(probs, axis=2)
    sy_sampled_ac = tf.cast(ac_indices, dtype=tf.int32)

    loss = tf.reduce_sum(tf.multiply(losses, sy_adv_n))
    update_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    print(losses.get_shape())
    supervised_log_loss = tf.reduce_sum(losses)
    supervised_update_op = tf.train.AdamOptimizer(learning_rate).minimize(supervised_log_loss)

    # ========================================================================================#
    #                           ----------SECTION 5----------
    # Optional Baseline - Defining Second Graph
    # ========================================================================================#

    if nn_baseline:
        baseline_prediction = tf.squeeze(build_mlp(
            sy_ob_no,
            1,
            "nn_baseline",
            n_layers=n_layers,
            size=size))
        # Define placeholders for targets, a loss function and an update op for fitting a
        # neural network baseline. These will be used to fit the neural network baseline.
        # YOUR_CODE_HERE
        sy_rew_n = tf.placeholder(shape=[None], name="rew", dtype=tf.int32)
        loss2 = tf.losses.mean_squared_error(labels=sy_rew_n, predictions=baseline_prediction)
        baseline_update_op = tf.train.AdamOptimizer(learning_rate).minimize(loss2)

    # ========================================================================================#
    # Tensorflow Engineering: Config, Session, Variable initialization
    # ========================================================================================#

    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)

    network_params = tf.trainable_variables()
    # saver = tf.train.Saver(network_params, max_to_keep=1)
    saver_global = tf.train.Saver(tf.global_variables())

    checkpoint_global_dir = os.path.join(os.curdir, 'PG_DISCRETE_200_GLOBAL')
    if not os.path.exists(checkpoint_global_dir):
        print('creating checkpoint_global directory..')
        os.makedirs(checkpoint_global_dir)
    model_global_prefix = os.path.join(checkpoint_global_dir, "model.ckpt")
    ckpt_global = tf.train.get_checkpoint_state(checkpoint_global_dir)

    if ckpt_global and tf.train.checkpoint_exists(ckpt_global.model_checkpoint_path):
        print("Reading global parameters from %s" % ckpt_global.model_checkpoint_path)
        saver_global.restore(sess, ckpt_global.model_checkpoint_path)

    # checkpoint_actor_dir = os.path.join(os.curdir, 'PG_DISCRETE_200')
    # if not os.path.exists(checkpoint_actor_dir):
    #     os.makedirs(checkpoint_actor_dir)
    # model_prefix = os.path.join(checkpoint_actor_dir, "model.ckpt")
    # ckpt_1 = tf.train.get_checkpoint_state(checkpoint_actor_dir)

    # if ckpt_1 and tf.train.checkpoint_exists(ckpt_1.model_checkpoint_path):
    #     print("Reading actor parameters from %s" % ckpt_1.model_checkpoint_path)
    #     saver.restore(sess, ckpt_1.model_checkpoint_path)

    uninitialized_vars = []
    for var in tf.global_variables():
        try:
            sess.run(var)
        except tf.errors.FailedPreconditionError:
            uninitialized_vars.append(var)

    if len(uninitialized_vars) > 0:
        init_new_vars_op = tf.variables_initializer(uninitialized_vars)
        sess.run(init_new_vars_op)
        print('Creating fresh model..')

    # print('saving model params to, ', model_global_prefix)
    # saver_global.save(sess, model_global_prefix)
    # ========================================================================================#
    # Training Loop
    # ========================================================================================#

    total_timesteps = 0
    t = 0
    bins = np.arange(start=-1, stop=1, step=1/disc_levels, dtype=np.float32)
    print('len(bins): ', len(bins))

    def testing():
        print('testing the existing model..')
        ob = env.reset()
        steps = 0
        done = False
        total_r = 0
        while not done:
            k = np.reshape(np.array(ob), newshape=(-1, len(ob)))
            ac_ind = sess.run(sy_sampled_ac, feed_dict={sy_ob_no: k})
            ac_ind = np.reshape(ac_ind, newshape=(ac_ind.shape[1]))
            ac = bins[ac_ind]
            if steps > 200:
                noise = np.random.normal(0.0, 0.000, ac_dim)
                ac += noise
            ob, rew, done, _ = env.step(ac.astype(np.float32))
            total_r += rew
            env.render()
            steps += 1
            if steps > max_path_length:
                break
        print('total steps: ', steps)
        print('total reward: ', total_r)
        return steps, total_r

    test = False
    if test:
        print('testing..')
        rewards = []
        for i in range(5):
            steps, reward = testing()
            rewards.append(reward)
        print(rewards)
        return

    exp = False
    if exp:
        print('generating exp data..')
        import pickle as pkl
        paths = []
        timesteps_this_batch = 0
        while True:
            ob_dict = env.reset(project=False)
            # ob = env.reset()
            obs, acs = [], []
            total_r = 0
            while True:
                ob = dict_to_list(ob_dict)
                ob_vec = dict_to_list_full(ob_dict)
                obs.append(ob_vec)
                k = np.reshape(np.array(ob), newshape=(-1, len(ob)))
                one_hot_ac = sess.run(sy_sampled_ac, feed_dict={sy_ob_no: k})
                ac_ind = np.reshape(one_hot_ac, newshape=(one_hot_ac.shape[1]))
                ac = bins[ac_ind]
                noise = np.random.normal(0.0, 0.06, ac_dim)
                ac += noise
                acs.append(ac)
                ob_dict, rew, done, _ = env.step(ac, project=False)
                # ob, rew, done, _ = env.step(ac.astype(np.float32))
                total_r += rew
                if done:
                    done = False
                    break
            path = {"observation": np.array(obs),
                    "action": np.array(acs)}

            if total_r > 0:
                timesteps_this_batch += len(path['action'])
                paths.append(path)

            print(timesteps_this_batch, total_r)
            if timesteps_this_batch > 10000:
                break
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_na = np.concatenate([path["action"] for path in paths])
        pkl.dump(ob_no, open('./simulation_data_1_1/obs_pg_full.p', 'wb'))
        pkl.dump(ac_na, open('./simulation_data_1_1/acts_pg_full.p', 'wb'))
        return

    def submit():
        from osim.http.client import Client
        remote_base = "http://grader.crowdai.org:1729"
        crowdai_token = "01342e360022c2def5c2cc04c5843381"
        Client = Client(remote_base)
        observation = Client.env_create(env_id="ProstheticsEnv", token=crowdai_token)
        while True:
            k = np.reshape(np.array(observation), newshape=(-1, len(observation)))
            ac_ind = sess.run(sy_sampled_ac, feed_dict={sy_ob_no: k})
            ac_ind = np.reshape(ac_ind, newshape=(ac_ind.shape[1]))
            action = bins[ac_ind]
            [observation, reward, done, info] = Client.env_step(action, True)
            if done:
                observation = Client.env_reset()
                if not observation:
                    break
        Client.submit()

    submission=True
    if submission:
        print('trying to submit..')
        submit()
        return

    def supervised_training(num_iter):
        _, best_rew = testing()
        print('supervised training..')
        import pickle as pkl
        train_obs = pkl.load(open('./simulation_data_1_1/obs_pg_100.p', 'rb'))
        train_acts = pkl.load(open('./simulation_data_1_1/acts_pg_100.p', 'rb'))
        batch_size = 500

        def get_batch_size(batch_size):
            num_examples = len(train_obs)
            ind = np.random.choice(num_examples, batch_size)
            x_batch, y_batch = train_obs[ind], train_acts[ind]
            return x_batch, y_batch

        for i in range(num_iter):
            obs_batch, act_batch = get_batch_size(batch_size)
            ind_act_batch = np.digitize(act_batch, bins) - 1
            act_batch = bins[ind_act_batch]
            print('updating model params..')
            sess.run(supervised_update_op, feed_dict={sy_ob_no: obs_batch, sy_ac_na: ind_act_batch})

            print('validating..')
            steps, rew = testing()
            if rew > best_rew:
                best_rew = rew
                print('saving the model to ', model_global_prefix)
                saver_global.save(sess, model_global_prefix)

    supervised_train = False
    if supervised_train:
        supervised_training(30)
        return

    for i in range(1):
        prev_steps, best_rew = testing()
    no_improvement = 0
    for itr in range(n_iter):
        print("********** Iteration %i ************" % itr)
        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        episode_steps = []
        episode = 0
        while True:
            ob = env.reset()
            obs, acs, rewards = [], [], []
            animate_this_episode = (len(paths) == 0 and (itr % 30 == 0) and animate)
            steps = 0
            ac_ind = np.digitize(env.action_space.sample(), bins)-1
            ac = bins[ac_ind]
            total_r = 0
            while True:
                if animate_this_episode:
                    env.render()
                    time.sleep(0.05)
                obs.append(ob)
                k = np.reshape(np.array(ob), newshape=(-1, len(ob)))
                # print('sampling an action...')
                if steps%1 == 0:
                    ac_ind = sess.run(sy_sampled_ac, feed_dict={sy_ob_no: k})
                    ac_ind = np.reshape(ac_ind, newshape=(ac_ind.shape[1]))
                    ac = bins[ac_ind]
                # print('getting observations from env ..')

                if steps > prev_steps-10:
                    noise = np.random.normal(0.0, 0.1, ac_dim)
                else:
                    noise = np.random.normal(0.0, 0.03, ac_dim)
                ac = ac + noise
                # else:
                #     noise = np.random.normal(0.0, 0.001, ac_dim)
                ac_ind = np.digitize(ac, bins) - 1
                acs.append(ac_ind)
                ob, rew, done, _ = env.step(ac)  # transition dynamics P(s_t+1/s_t,a_t), r(s_t+1/s_t,a_t)
                total_r += rew
                if rew > 0:
                    rew = rew*10
                else:
                    rew = rew*10
                rewards.append(rew)
                steps += 1
                if done or steps > max_path_length:
                    break
            if episode < 2:
                path = {"observation": np.array(obs),
                        "reward": np.array(rewards),
                        "action": np.array(acs)}
            else:
                path = {"observation": np.array(obs[-20:]),
                        "reward": np.array(rewards[-20:]),
                        "action": np.array(acs[-20:])}
            episode += 1
            episode_steps.append(steps)
            if total_r > 0:
                paths.append(path)
                timesteps_this_batch += pathlength(path)
                print(total_r)
            if timesteps_this_batch > min_timesteps_per_batch:
                break

        total_timesteps += timesteps_this_batch
        episode_steps = np.array(episode_steps)
        prev_steps = int(np.mean(episode_steps))

        # Build arrays for observation, action for the policy gradient update by concatenating
        # across paths
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_na = np.concatenate([path["action"] for path in paths])
        ac_na = ac_na.reshape([-1, ac_dim])
        # ====================================================================================#
        #                           ----------..----------
        #                             Computing Q-values
        # ====================================================================================#

        # DYNAMIC PROGRAMMING
        if reward_to_go:
            q_n = list()
            for path in paths:
                pLen = pathlength(path)
                q_p = np.zeros(pLen)
                q_p[pLen - 1] = path['reward'][pLen - 1]
                for t in reversed(range(pLen - 1)):
                    q_p[t] = path['reward'][t] + gamma * q_p[t + 1]
                q_p = np.array(q_p)
                q_n.append(q_p)
        else:
            q_n = list()
            for path in paths:
                pLen = pathlength(path)
                q_p = 0
                for t in range(pLen):
                    q_p = q_p + (gamma ** t) * (path['reward'][t])
                q_n.append(q_p * np.ones(pLen))
        q_n = np.concatenate(q_n)

        # ====================================================================================#
        #                           ----------SECTION 5----------
        # Computing Baselines
        # ====================================================================================#

        if nn_baseline:
            b_n = sess.run(baseline_prediction, feed_dict={sy_ob_no: ob_no})
            b_n = normalize(b_n, np.mean(q_n), np.std(q_n))
            adv_n = q_n - b_n
        else:
            adv_n = q_n.copy()

        # ====================================================================================#
        #                           ----------SECTION 4----------
        # Advantage Normalization
        # ====================================================================================#

        if normalize_advantages:
            print('normalize: ', normalize_advantages)
            adv_n = normalize(adv_n)

        # ====================================================================================#
        #                           ----------SECTION 5----------
        # Optimizing Neural Network Baseline
        # ====================================================================================#
        if nn_baseline:
            print('baseline: ', nn_baseline)
            sess.run(baseline_update_op, feed_dict={sy_ob_no: ob_no, sy_rew_n: q_n})

        # ====================================================================================#
        #                           ----------SECTION 4----------
        # Performing the Policy Update
        # ====================================================================================#

        t += 1

        print("action_ph_shape", ac_na.shape)
        print("obs_ph_shape", ob_no.shape)
        print('adv_ph_shape', adv_n.shape)

        for i in range(1):
            print('updating model params..')
            sess.run(update_op, feed_dict={sy_ac_na: ac_na, sy_ob_no: ob_no, sy_adv_n: adv_n})

            _, new_ret = testing()
            if new_ret > best_rew:
                print('saving model params to, ', model_global_prefix)
                saver_global.save(sess, model_global_prefix)
                best_rew = new_ret
            else:
                no_improvement += 1

        # if no_improvement > 10:
        #     supervised_training(10)
        # Log diagnostics

        returns = [path["reward"].sum() for path in paths]
        ep_lengths = [pathlength(path) for path in paths]
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
    parser.add_argument('--exp_name', type=str, default='desc_200_1_1')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--n_iter', '-n', type=int, default=1000000)
    parser.add_argument('--batch_size', '-b', type=int, default=500)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-6)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true', default=True)
    parser.add_argument('--test', '-t', action='store_true', default=False)
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true', default=True)
    parser.add_argument('--nn_baseline', '-bl', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=3)
    parser.add_argument('--size', '-s', type=int, default=150)
    parser.add_argument('--disc_levels', '-dl', type=int, default=100)
    args = parser.parse_args()

    print('test: ', args.test)
    if not (os.path.exists('data')):
        os.makedirs('data')
    logdir = args.exp_name+'_'+time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    env = ProstheticsEnv(visualize=False, integrator_accuracy=1e-4)
    env.change_model(model='3D', difficulty=2, prosthetic=True, seed=0)
    print('ac_dim: ', env.action_space.shape)
    print('obs_dim: ', env.observation_space.shape)
    print('normalize: ', not (args.dont_normalize_advantages))

    max_path_length = args.ep_len if args.ep_len > 0 else None
    train_PG(
        exp_name=args.exp_name,
        env_name=env,
        n_iter=args.n_iter,
        gamma=args.discount,
        min_timesteps_per_batch=args.batch_size,
        max_path_length=max_path_length,
        learning_rate=args.learning_rate,
        reward_to_go=args.reward_to_go,
        animate=args.render,
        logdir=os.path.join(logdir, '%d' % 0),
        seed=0,
        normalize_advantages=not (args.dont_normalize_advantages),
        nn_baseline=args.nn_baseline,
        n_layers=args.n_layers,
        size=args.size,
        test=args.test,
        disc_levels = args.disc_levels
    )


if __name__ == "__main__":
    main()
