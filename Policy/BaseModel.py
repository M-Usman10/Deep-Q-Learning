
from Policy.callbacks import get_tensorboard_callback, get_checkpoint_call_back


class BaseModel:
    def __init__(self, input_shape, num_outputs, optimizer, prefix):
        """
        params:
            input_shape: a list that specifies input_shapes 
                            of all the inputs to the model (batch dim not included)
            num_outputs:num of neurons in the last layer
            optimizer: optimizer function used during training
            prefix: name of the model (this name is appended to the checkpoints and logs)
                        
        """
        
        self.input_shape = input_shape
        self.num_outputs = num_outputs
        self.optimizer = optimizer
        self.prefix = prefix
        self.Model = None
        
    def construct_model(self):
        """
        Model is constructed here and self.Model stores the link to the Model
        """
        raise Exception('Unimplemented Error ')

    def train_generator(self, generator, epochs=10, log_dir="../../logs", checkpoint_dir="../../checkpoints",
                        save_n_epochs=10):
        """

        :param generator: gives a list [inputs,targets] that is fed into the model
        :param epochs: num of epochs to train the model
        :param log_dir: directory where to place logs
        :param checkpoint_dir: directory in which to save checkpoints
        :param save_n_epochs: save model weights after N epochs
        :return:
        """
        tensorboard = get_tensorboard_callback(logdir=log_dir+"/"+self.prefix)
        checkpoint = get_checkpoint_call_back(checkpoint_dir,self.prefix+"_checkpoint_", period=save_n_epochs)
        
        self.Model.fit_generator(generator=generator,
                                 steps_per_epoch=len(generator),
                                 epochs=epochs,
                                 verbose=1,
                                 max_queue_size=3,
                                 callbacks=[tensorboard, checkpoint])

    def train(self, train_data_x, train_data_y, val_data_x, val_data_y, epochs=10, log_dir="../../logs",
              checkpoint_dir="../../checkpoints", save_n_epochs=10):
        """

        :param train_data_x: list of training inputs ,shape of each input=(numexamples,numfeatures)
        :param train_data_y: list of training outputs
        :param val_data_x: list of validation inputs ,shape of each input=(numexamples,numfeatures)
        :param val_data_y: list of validations outputs
        :param epochs:  num of epochs
        :param log_dir:  directory where to place logs
        :param checkpoint_dir: directory in which to save checkpoints
        :param save_n_epochs: save model weights after N epochs
        :return:
        """
        tensorboard = get_tensorboard_callback(logdir=log_dir + "/" + self.prefix)
        checkpoint = get_checkpoint_call_back(checkpoint_dir, self.prefix+"_checkpoint_", period=save_n_epochs)
        
        self.Model.fit(x=train_data_x,
                       y=train_data_y,
                       validation_data=(val_data_x, val_data_y),
                       batch_size=64,
                       epochs=epochs,
                       verbose=1,
                       callbacks=[tensorboard,checkpoint])
        
    def predict(self, observation):
        """

        :param observation:
        :return:
        """
        
        raise Exception('Unimplemented Error ')
        