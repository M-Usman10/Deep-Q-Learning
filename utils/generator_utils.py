import numpy as np


def preprocess_rgb_img(img, Frame_Processor):
    return Frame_Processor.process_frame(img)


def softmax(logits):
    logits-=np.max(logits)
    return np.exp(logits)/np.sum(np.exp(logits))

def explore_boltzman( observation,gain,Policy_Network,actions):
        observation = np.stack(observation, axis=-1)
        observation = observation.reshape(1, *observation.shape)
        q_values = Policy_Network.Model.predict(observation)[0]
        logits = q_values / gain
        prob = softmax(logits)
        return np.random.choice(actions, p=prob)
def explore_eps_greedy(observation,eps,Policy_Network,actions):
    observation = np.stack(observation, axis=-1)
    observation = observation.reshape(1, *observation.shape)
    q_values = Policy_Network.Model.predict(observation)[0]
    ind=np.argmax(q_values)
    prob=np.ones(actions.shape)*eps
    prob[ind]=1-eps
    return np.random.choice(actions, p=prob)