import numpy as np
from activation import *
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


class RNNCell(object):
    """RNN Cell class."""

    def __init__(self, input_size, hidden_size):

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Activation function for
        self.activation = Tanh()

        # hidden dimension and input dimension
        h = self.hidden_size
        d = self.input_size

        # Weights and biases
        self.W_ih = np.random.randn(h, d)
        self.W_hh = np.random.randn(h, h)
        self.b_ih = np.random.randn(h)
        self.b_hh = np.random.randn(h)

        # Gradients
        self.dW_ih = np.zeros((h, d))
        self.dW_hh = np.zeros((h, h))

        self.db_ih = np.zeros(h)
        self.db_hh = np.zeros(h)

    def init_weights(self, W_ih, W_hh, b_ih, b_hh):
        self.W_ih = W_ih
        self.W_hh = W_hh
        self.b_ih = b_ih
        self.b_hh = b_hh

    def zero_grad(self):
        d = self.input_size
        h = self.hidden_size
        self.dW_ih = np.zeros((h, d))
        self.dW_hh = np.zeros((h, h))
        self.db_ih = np.zeros(h)
        self.db_hh = np.zeros(h)

    def __call__(self, x, h):
        return self.forward(x, h)

    def forward(self, x, h):
        """
        RNN Cell forward (single time step).

        Input (see writeup for explanation)
        -----
        x: (batch_size, input_size)
            input at the current time step

        h: (batch_size, hidden_size)
            hidden state at the previous time step and current layer

        Returns
        -------
        h_prime: (batch_size, hidden_size)
            hidden state at the current time step and current layer
        """

        """
        ht = tanh(Wihxt + bih + Whhht−1 + bhh) 
        """
        h_prime = self.activation.forward(np.dot(x, self.W_ih.T) + self.b_ih + np.dot(h, self.W_hh.T) + self.b_hh) # TODO
        """ np.dot automatically takes care of tensor to numpy array conversion"""
        return h_prime
        # raise NotImplementedError

    def backward(self, delta, h, h_prev_l, h_prev_t):
        """
        RNN Cell backward (single time step).

        Input (see writeup for explanation)
        -----
        delta: (batch_size, hidden_size)
                Gradient w.r.t the current hidden layer

        h: (batch_size, hidden_size)
            Hidden state of the current time step and the current layer

        h_prev_l: (batch_size, input_size)
                    Hidden state at the current time step and previous layer

        h_prev_t: (batch_size, hidden_size)
                    Hidden state at previous time step and current layer

        Returns
        -------
        dx: (batch_size, input_size)
            Derivative w.r.t.  the current time step and previous layer

        dh: (batch_size, hidden_size)
            Derivative w.r.t.  the previous time step and current layer

        """
        batch_size = delta.shape[0]
        # 0) Done! Step backward through the tanh activation function.
        # Note, because of BPTT, we had to externally save the tanh state, and
        # have modified the tanh activation function to accept an optionally input.

        # print(delta.shape)
        # print(self.activation.derivative(h).shape)
        dz = self.activation.derivative(h) * delta# TODO
        # print(self.activation.derivative(h).shape)

        # 1) Compute the averaged gradients of the weights and biases
        # print(h_prev_l.shape)
        # print(dz.shape)
        self.dW_ih += (1/batch_size)*(np.dot(dz.T,h_prev_l)) # TODO
        self.dW_hh += (1/batch_size)*(np.dot(dz.T, h_prev_t)) # TODO
        self.db_ih += (1/batch_size)*np.sum(dz, axis=0) # TODO
        self.db_hh += (1/batch_size)*np.sum(dz, axis=0) # TODO

        # # 2) Compute dx, dh
        # print(self.W_ih.shape)
        # print(self.W_hh.shape)
        dx = np.dot(dz, self.W_ih) # TODO
        dh = np.dot(dz, self.W_hh) # TODO
        # print(self.W_hh.shape)

        # 3) Return dx, dh
        return dx, dh
        # raise NotImplementedError
