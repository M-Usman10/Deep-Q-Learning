import numpy as np
import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras import backend as KB
from keras.callbacks import Callback


def get_tensorboard_callback(logdir='../logs/fc_net'):
    """
    Params: 
        logdir: where to write the tensorboard logs
    """
    return TensorBoard(log_dir=logdir,
                          write_graph=True)

def get_checkpoint_call_back(checkpoint_dir="../checkpoints",prefix="fc_net_chkpoint",period=1):
    """
    Params:
        checkpoint dir : the directory where to save the checkpoints
        prefix: prefix for the filename
        period: after how many epochs to store checkpoints
        
    """
    path=checkpoint_dir+"/"+prefix+"_{epoch:02d}.hdf5"
    checkpoint=ModelCheckpoint(path, monitor='loss',
                                   verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=period)
    return checkpoint
        