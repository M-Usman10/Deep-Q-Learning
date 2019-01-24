import numpy as np

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
    prob=(np.ones(actions.shape)*eps)/(actions.shape[0]-1)
    prob[ind]=1-eps
    return np.random.choice(actions, p=prob)