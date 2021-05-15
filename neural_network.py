"""
Neural network implementation.


The neural network (NN) represents the policy of the car.
A Reinforcement learning (RL) stategy is used to train the NN which is based on an
evolutionary model.


@author: Damian Hoedtke
@date: May, 2021

"""
import numpy as np
from numpy import array
from numpy.random import rand

class NN(object):
    """ Neural Network class. """

    def __init__(self, n_inputs, n_outputs, hidden_layers, p_mutation, scale, learning_scale, weights=None):

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.p_mutation = p_mutation
        self.scale = scale
        self.learning_scale = learning_scale

        if len(hidden_layers) < 1:
            raise Exception("Too less hidden_layers given. \nMinimum required is 1.")
        self.hidden_layers = hidden_layers

        # initial weights should be small
        if weights == None:

            self.weights = []

            # first weights
            self.weights.append( self.scale * (rand(hidden_layers[0], self.n_inputs) - 0.5) )

            for i in range(len(hidden_layers) - 1):
                self.weights.append( self.scale * (rand(hidden_layers[i+1], hidden_layers[i]) - 0.5) )

            self.weights.append( self.scale * (rand(self.n_outputs, hidden_layers[-1]) - 0.5) )
        else:
            self.weights = weights

        return

    def get_child(self, fertilizer):

        # mutation of weights with probability p_mutation
        for w, v in zip(self.weights, fertilizer.weights):
            rands = np.random.rand(*w.shape)
            mutated = rands < self.p_mutation
            mutated = array(mutated, dtype='int')
            mutation = self.learning_scale * self.scale * (np.random.rand(*w.shape) - 0.5)

            #w *= 0.8
            #w += 0.2 * v
            w += np.multiply(mutated, mutation) # add mutation

        return NN(self.n_inputs, self.n_outputs, self.hidden_layers,
                  self.p_mutation, self.scale, self.learning_scale, self.weights)

    def predict(self, data):
        # check format
        if self.n_inputs != data.size:
            raise Exception("Wrong data shape for NN prediction.")

        for w in self.weights:
            #print(w.shape)
            data = w @ data

        # check output format
        if self.n_outputs != data.size:
            raise Exception("Wrong data shape after NN prediction.")

        return data

    # END
