from utils.BaseModel import BaseModel
from utils.tools import construct_model_from_csv
import tensorflow as tf
import keras.layers as layers
from keras.models import Model as keras_model
from keras import backend as KB
import numpy as np
import keras.optimizers as KO
class DQN(BaseModel):
   def __init__(self, input_shape, num_outputs, optimizer, arch_config_file):
       """
       params:
           input_shape:specifies input_shape
                           for the inputs to the model (batch dim not included)
           optimizer: optimizer function used during training
           arch_config_files:path of file that specify the design of network
       """
       BaseModel.__init__(self, input_shape, num_outputs, optimizer, "DQN")
       self.arch_config_file = arch_config_file



   def compute_pred_q_a(self,pred_qvals,actions):


       pred_Q=KB.sum(pred_qvals*actions, axis=-1,keepdims=True)

       return pred_Q



   def construct_model(self,training=True):
       """
       Model is constructed here and self.Model stores the link to the Model
       """
       state_inputs = layers.Input(shape=(self.input_shape))

       q_values = construct_model_from_csv(self.arch_config_file,state_inputs)
       assert q_values.shape[-1] == self.num_outputs,"outputs in last layer should \
                                                       be equal to the number of actions allowed"
       if training:
           action_one_hot_inputs=layers.Input(shape=(self.num_outputs,))
           pred_q_a=layers.Lambda(lambda x:self.compute_pred_q_a(*x))([q_values,action_one_hot_inputs])


           self.Model = keras_model(inputs=[state_inputs,action_one_hot_inputs], output=pred_q_a)
           self.Model.compile(loss="mse", optimizer=self.optimizer)

       else:
           self.Model = keras_model(inputs=[state_inputs], output=q_values)

       self.Model.summary()
       self.training_model=training

   def predict(self,states):

       assert (len(states.shape)==4),"states ndim not valid for predicting. Expected ndims =4 and received {} ".format(len(states.shape))
       qvals=0
       if self.training_model:
         qvals=self.Model.predict([states,np.zeros((states.shape[0],self.num_outputs))])[0]
       else:
         qvals=self.Model.predict(states)
       return qvals
   def load_weights(self):
       self.Model.load_weights('weights.h5')