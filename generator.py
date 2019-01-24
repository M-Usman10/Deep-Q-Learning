from Exploration import *
from Policy.PolicyNetwork import DQN
from keras.optimizers import Adam
from collections import deque
import tensorflow as tf
import os
import gym
from utils.Preprocessor import Frame_Processor
from utils.tools import share_gpu
class DataGenerator:
    def __init__(self,config):
        """
        Parallelised Data Generator to fill replay buffer
        :param config: configurations loaded from config.yaml
        """
        self.config = config
        self.gamma = self.config['gamma']
        self.decay = self.config['decay']
        self.gain = self.config['gain']
        self.eps = self.config['eps']
        self.max_ep_steps = self.config['max_episode_steps']
    def init_unpickleable_objects(self,shared_objs):
        """
        UnpickleAble objects can't be set in generator as parallelism is not possible with that
        :param shared_objs: Shared Queue and Shared Variable
        :return: NOthing
        """
        self.env = gym.make(self.config['env_name'])
        self.actions = np.arange(0,self.env.action_space.n,dtype=int)
        self.model = DQN(input_shape=self.config['input_shape'], num_outputs=self.config['actions'],
                        lr=self.config['learning_rate'],
                        arch_config_file=os.path.join(self.config['path'], "qnetwork.csv"))
        self.model.construct_model(training=False)
        self.FP=Frame_Processor(crop=True,convert_to_grayscale=True,normalize=True,coords=self.config,resize=True,resize_y=110,resize_x=84)
        self.buffer = shared_objs['replay_buffer']
        self.load_network = shared_objs['load_network']


    def generate_data(self,shared_objs):
        """
        Interact with environment applies specified exploration policy and then stores data into a shared replay_buffer
        :param shared_objs: Shared Queue and Shared Variable
        :return: Nothing
        """
        self.init_unpickleable_objects(shared_objs)
        if (self.config['gpu']==True):
            share_gpu()  # Allows keras to share gpu
        print ('Started Generation Process')
        obs_queue=deque(maxlen=self.config['nstack'])  # stacking multiple images(or observations) to pass into NN
        ite,ite2,log_dir,done=0,0,'tf_logs',True
        # Placeholders for tenserboard
        reward_placeholder = tf.placeholder(tf.float32, shape=(), name='Reward')
        total_q_placeholder=tf.placeholder(tf.float32, shape=(), name='Q')
        reward_summary = tf.summary.scalar('Reward', reward_placeholder)
        q_summary = tf.summary.scalar('Reward', total_q_placeholder)
        file_writer=tf.summary.FileWriter(log_dir, tf.get_default_graph())
        episode_reward,episode,total_q=0,0,0
        # Generate Data Unless process is killed (or sigint is passed)
        while True:
            with self.load_network.get_lock():  # Load weights for Newly saved target Network
                if self.load_network.value == 1:
                    self.model.load_weights()
                    print("New Network Loaded")
                    self.load_network.value=0
                    ite += 1
                    self.gain *= 1 / (1 + self.decay * ite)
                    self.eps *= 1 / (1 + self.decay * ite)
            #  Reset Env and prints Reward when episode is done
            if done or ite2 > self.max_ep_steps:
                ite2 += 1e-5
                print('Steps Completed in current episode:{} with reward {} and q {} and mean r {} and mean q {}'
                      .format(ite2,episode_reward,total_q,episode_reward/ite2,total_q/ite2))
                obs = self.env.reset()
                obs = self.FP.process_frame(obs)
                total_q = 0
                with tf.Session() as sess:
                    summary_str = reward_summary.eval(feed_dict={reward_placeholder:episode_reward})
                    file_writer.add_summary(summary_str, episode)
                    summary_str = reward_summary.eval(feed_dict={reward_placeholder: total_q/ite2})
                    file_writer.add_summary(summary_str, episode)
                for i in range(self.config['nstack']):
                    obs_queue.append(obs)
                episode_reward = 0
                episode+=1
                ite2 = 0
            if self.config['exploration']=='greedy':
                action = explore_eps_greedy(obs_queue, self.eps, self.model, self.actions)
            elif self.config['exploration']=='boltzman':
                action = explore_boltzman(obs_queue, self.gain, self.model, self.actions)
            observation, reward, done, info=self.env.step(action)
            episode_reward+=reward
            obs = self.FP.process_frame(observation)
            obs_queue.append(obs)
            observation = np.stack(obs_queue, axis=-1)
            observation = observation.reshape(1, *observation.shape)
            q_values = self.model.Model.predict(observation)[0]
            true_reward=reward+self.gamma*np.max(q_values)
            total_q+=true_reward
            self.buffer.put((observation,true_reward,action,done),block=True,timeout=None)
            ite2+=1