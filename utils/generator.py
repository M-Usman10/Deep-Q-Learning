from utils.generator_utils import *
from utils.PolicyNetwork import DQN
from keras.optimizers import Adam
from collections import deque
import tensorflow as tf
import gym
def generate(config):
        print ('Started Process')
        env = gym.make(config['env_name'])
        input_shape=(110,84,4)#env.obs_shape
        num_outputs=env.action_space.n#env.num_actions
        optimizer=Adam(lr=0.0001)
        design_config_file=r"utils\qnetwork.csv"
        model=DQN(input_shape=input_shape,num_outputs=num_outputs,
                optimizer=optimizer,arch_config_file=design_config_file)
        model.construct_model(training=False)
        config['policy']=model
        gamma=config['gamma']
        actions=np.arange(0,env.action_space.n,dtype=int)
        obs_queue=deque(maxlen=config['nstack'])
        buffer=config['replay_buffer']
        decay=config['decay']
        FP=config['Frame_Processor']
        saved_new=config['saved_new']
        max_ep_steps=config['max_episode_steps']
        done=True
        gain=config['gain']
        ite=0
        ite2 = 0
        log_dir = 'tf_logs'
        reward_placeholder = tf.placeholder(tf.float32, shape=(), name='Reward')
        reward_summary = tf.summary.scalar('Reward', reward_placeholder)
        file_writer=tf.summary.FileWriter(log_dir, tf.get_default_graph())
        episode_reward=0
        episode=0
        print ('Loop Started')
        while True:
            with saved_new.get_lock():
                if (saved_new.value==1):
                    config['policy'].load_weights()
                    print ("New Network Loaded")
                    saved_new.value=0
            if done or ite2>max_ep_steps:
                print('Steps Completed in current episode:{} wirh reward {}'.format(ite2,episode_reward))
                ite2=0
                obs=env.reset()
                obs=FP.process_frame(obs)
                reward=0
                with tf.Session() as sess:
                    summary_str = reward_summary.eval(feed_dict={reward_placeholder:episode_reward})
                    file_writer.add_summary(summary_str, episode)
                for i in range(config['nstack']):
                    obs_queue.append(obs)
                episode_reward = 0
                episode+=1
            action = explore_boltzman(obs_queue, gain, config['policy'], actions)
            observation, reward, done, info=env.step(action)
            episode_reward+=reward
            obs = FP.process_frame(observation)
            obs_queue.append(obs)
            observation = np.stack(obs_queue, axis=-1)
            observation = observation.reshape(1, *observation.shape)
            q_values = config['policy'].Model.predict(observation)[0]
            true_reward=reward+gamma*np.max(q_values)
            buffer.put((observation,true_reward,action,done),block=True,timeout=None)
            gain *= 1 / (1 + decay * ite)
            ite2+=1
            ite+=1