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
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU, PReLU

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
                #if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
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
    #max_steps = args.max_timesteps or env.spec.timestep_limit
    max_steps=500
    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    mean_ret=[]
    
    xTr=aggregated_data['observations']
    yTr=aggregated_data['actions']
    yTr=np.reshape(yTr,(yTr.shape[0],yTr.shape[2]))
    #construct Model
    with tf.Session():
        tf_util.initialize()
        model = Sequential()
        model.add(Dense(120, input_dim=xTr.shape[1], init="uniform",
            activation="linear"))
        model.add(LeakyReLU(alpha=.01))
        model.add(Dropout(0.5))
        model.add(Dense(100, init="uniform", activation="linear"))
        model.add(LeakyReLU(alpha=.01))
        model.add(Dropout(0.5))
        model.add(Dense(80, init="uniform", activation="linear"))
        model.add(LeakyReLU(alpha=.01))
        model.add(Dropout(0.5))
        model.add(Dense(yTr.shape[1]))
        
        #compile Model
        model.compile(loss='mean_squared_error',
                      optimizer='adam',
                      metrics=['accuracy'])
        
        model.fit(xTr, yTr,
                      epochs=30,
                      batch_size=1000,validation_split=0.1) 
        model.save(r"C:\Users\DELL\Desktop\GITHUB\DeepReinforcementLearning\1_BehaviourCloning_DAGGER\Policies\policyDNN_ANT_DA.h5")
        for i in range(10):
            xTr=aggregated_data['observations']
            yTr=aggregated_data['actions']
            yTr=np.reshape(yTr,(yTr.shape[0],yTr.shape[2]))
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
                #if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)
    
            print('returns', returns)
            aggregated_data = {'observations': np.concatenate((aggregated_data['observations'],
                             np.array(observations))),'actions': np.concatenate((aggregated_data['actions'],
                             np.array(actions))) }
            model = load_model(r"C:\Users\DELL\Desktop\GITHUB\DeepReinforcementLearning\1_BehaviourCloning_DAGGER\Policies\policyDNN_ANT_DA.h5")
            model.fit(xTr, yTr,
                      epochs=30,
                      batch_size=1000,validation_split=0.1)
    return returns

    
    

if __name__ == '__main__':
    expert_data,args=main()
    returns=dagger(expert_data,args)
    
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(range(len(yAntDA50k)),yAntDA50k, linestyle='dashed', marker='.',
         markerfacecolor='green', markersize=12, label='DAGGER')
    plt.plot(range(len(yExp)),yExp, linestyle='dashed', marker='.',
         markerfacecolor='red', markersize=12, label='Expert Policy')
    plt.plot(range(len(yBC)),yBC, linestyle='dashed', marker='.',
         markerfacecolor='blue', markersize=12, label='Behaviour Clonning')
    
    plt.xlabel("Iterations")
    plt.ylabel("Reward")
    plt.grid()
    plt.legend()
    plt.show()
    '''

'''
y1=[51.28679953413875, 56.099106740947036, 51.60155531946228, 49.48939795530981, 52.815264026387915, 53.074269146800155, 53.217219242820626, 54.626407591389615, 51.2742886787491, 50.28811191450817, 52.876207596909055, 58.84379428628775, 54.782028254163876, 71.59347080905934, 60.278245089664786, 76.10909740706629, 86.85786570504078, 76.84162203361869, 91.08158093270353, 99.77519425498659, 113.97718464251439, 125.7879148542972, 102.4140882866472, 107.10871379902396, 104.96727560284887, 125.8450523621245, 94.5985348270908, 89.52716926529435, 83.6782661613109, 92.5795917571608, 83.69160453243198, 73.24119737119578, 95.40214341357918, 184.65329974618794, 104.14980914189726, 108.10352857339548, 133.55239947698848, 141.23673938245923, 157.35389571270773, 160.61517363237778, 220.97328092061468, 222.55273918170045, 220.92685025465173, 221.29821242501276, 217.36106197040397, 221.9826941011824, 222.3266436404757, 196.40267720320932, 190.49934855258948, 182.67394885598068, 180.84215171668373, 180.12441876796566, 191.00122714557187, 195.72482778373296, 181.36873705392006, 247.00320386529182, 241.822786782934, 304.660076820765, 208.0704470834124, 208.94840688687825, 229.8729070425502, 212.0093665387604, 218.96859401250134, 210.94950325007713, 209.95218931337732, 205.29247490432516, 208.58235389245144, 1148.2832109823983, 208.76665308607352, 1125.2704856448624, 1759.3190205271892, 1497.466090562436, 1033.0612776300359, 1276.7600536823845, 1724.4133801749745, 1362.1553975217555, 1175.0954755068608, 1305.4580706041656, 1014.6618616213374, 1125.7329842308618, 1045.4939897273584, 866.2319767900086, 1183.002603197618, 1186.66801973811, 996.2760334579082, 1184.0528742579722, 1211.1286854572008, 1299.6068794799337, 1336.1528721533452, 1309.0549439571269, 1669.9390215358947, 1354.109241376571, 1281.5024867984623, 1367.6566877527441, 1746.561456262936, 1707.3455246796602, 1677.5850538388072, 1820.262560187033, 1695.4979344158426, 1803.644442107657]
y2=[60.012291897764065, 60.15172817223024, 97.78301611599068, 68.58186171067577, 75.03376735481855, 58.331157544433836, 55.72633010034349, 59.10931839227771, 95.83430801814868, 137.84664027932328, 94.90278881577235, 124.3845383772149, 77.01335085832181, 150.53915627874645, 152.9446096505178, 175.95321524732847, 168.70426886511422, 200.40177861735066, 219.36744053424266, 202.68826410485062, 147.94205198084148, 142.9901602193305, 109.41105964846157, 97.04488988702245, 88.00784161904015, 83.1454635215327, 82.75082153767983, 71.61399618385066, 74.2877803760792, 72.45300800784288, 79.23708656429694, 76.53100719000531, 75.25806603296414, 80.83825526978444, 78.16245286534314, 80.00106555527142, 88.76273540978126, 98.28612014741525, 99.59663721222547, 216.57465415920606, 213.31586171360885, 219.38498764383476, 216.3851459813369, 221.48484044018983, 216.44408320507017, 217.60958921709405, 219.01942888088, 220.51168858524494, 218.24333284506994, 215.9477673475083, 218.56191493482277, 218.82557673349348, 217.87625390888962, 227.34565944637808, 291.0625790006954, 324.3397457856642, 485.18045024546245, 540.0481201633044, 615.5341110702421, 622.2137648977123, 637.6473120208327, 673.8438595734801, 658.2685723701024, 680.6505319261628, 684.7916536742136, 681.3826775808902, 695.9858223440891, 656.4521537947767, 677.7432502842195, 677.0432213694686, 676.0195749600063, 677.3858071060415, 669.0364863764786, 707.3913582134174, 676.3596444680074, 627.5379276891022, 660.4916674774605, 577.8401346892977, 644.7345834453322, 637.8933920809015, 762.4483639085117, 639.7957588312486, 647.4329120684741, 585.0700753173021, 673.646490482827, 529.923260212984, 611.3031906264922, 768.6908976144691, 531.7950121023944, 591.1729382746225, 626.2845090106508, 562.351238157887]
y3=[49.989462657327095, 50.104416320624175, 49.58863217314325, 48.13687347553083, 50.63997506239812, 48.201356688804665, 49.996389329652594, 51.69842596241468, 49.98332866799724, 48.26475750175602, 49.79445323899248, 52.858300832644304, 60.826130365484204, 57.93106257524057, 57.37916799177557, 69.48150978674543, 75.74328607165785, 75.76298106106114, 87.97143394458948, 90.17157024428256, 93.21221070449053, 133.96836396749848, 159.910742581733, 127.93502521449865, 104.34846284205818, 128.50378612947972, 95.00967881478773, 91.32468907254218, 88.77322413728875, 80.5917570379761, 77.12947972613146, 78.99024903039317, 110.73083668073777, 95.47686452266827, 96.0966152537265, 96.88572455798962, 153.00838879008379, 163.7938567407006, 89.41824380488254, 207.16096571741218, 159.37644280324605, 220.00154538501354, 217.8964732111155, 215.1331717178774, 219.4252026182348, 219.89331058717184, 223.80768236977778, 251.47517093168776, 237.35833591093566, 193.67749332083963, 196.41105893286183, 188.38952714102908, 195.07058909831844, 190.21345777159738, 196.677042936002, 249.49651133741224, 226.89284997062265, 200.2915106205832, 226.20477505318664, 675.2081718048065, 201.7552161943102, 219.65708924815857, 206.70195941247349, 210.55838652412154, 208.23341088797804, 1657.4457299557416, 212.7260543217056, 208.36698392552967, 1145.1496040656941, 1144.7037086358625, 1138.0951660114818, 856.0895756570566, 1087.8841867673527, 884.9320741820699, 1229.7114497232094, 959.9341925814905, 1141.8370105905094, 1184.0502889361442, 1110.1357598472432, 1172.9850827072216, 1792.3137412188971, 1417.6992466814252, 1483.5171037239857, 1399.472530683232, 1792.1047516144738, 1698.920081703879, 1830.6012487556463, 1704.8806505900884, 1793.9713831764604, 1663.9398004858506, 1725.6369002871502, 1662.5103886935756, 1778.3864681008288, 1667.4002121122994, 1387.1191134071162, 1726.7873761254816, 1733.1317040153892, 1727.5711810545913, 1700.0458547099695, 1667.344833799704]
yAntDA50k=[1254.11914350978,
 1683.8771063075876,
 3106.7165085646348,
 3883.733672413355,
 3534.196958577549,
 3404.6410053521067,
 4216.285919775204,
 4455.878713174183,
 4411.453745929789,
 4325.731255113232]

yExp=[4769.15]*10
yBC=[3898.34]*10
'''