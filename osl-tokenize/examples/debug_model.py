
#####################################

import sys
import numpy as np
from tensorflow.keras import layers
import tensorflow as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors

tf.random.set_seed(0)
np.random.seed(0)

dev_dir = '/Users/woolrich/dev'
osl_tokenize_dir = f'{dev_dir}/projects/osl-tokenize'
sys.path.append(osl_tokenize_dir)

from osl_tokenize.models import conv as tokenize
import osl_tokenize.layers as tokenize_layers

class TokenWeightsAmpLayer(layers.Layer):

    '''
    Layer to calculate the token amps 
    
    Parameters
    ----------
    n_params_out : int
        Number of parameters to output
    activation : str
        Activation function for the dense layer
    name : str
        Name of the layer
    
    '''

    def __init__(self, n_params_out, n_params_amp_out, activation="linear", name=None, **kwargs):
        super().__init__(**kwargs)

        self.n_params_out = n_params_out
        self.n_params_amp_out = n_params_amp_out

        self.dense_weights_layer = layers.Dense(n_params_out,
                                 activation="linear",
                                 name=name)
        self.norm_weights_layer = layers.LayerNormalization(center=True, scale=True)
        self.activation_weights_layer = layers.Activation(activation)

        ##

        self.dense_amp_layer = layers.Dense(n_params_amp_out,
                                 activation="linear",
                                 name=name)
        self.norm_amp_layer = layers.LayerNormalization(center=True, scale=True)
        self.activation_amp_layer = layers.Activation(activation)

        self.temperature = tf.Variable(tf.ones([1,]), trainable=False)

        res = 1.5/(self.n_params_amp_out-1)
        self.amp_lookup = tf.range(0.5, 0.5+self.n_params_amp_out*res, res, dtype=tf.float32)

    def call(self, inputs, **kwargs):
        ell = self.activation_weights_layer(self.dense_weights_layer(inputs))
        ell = self.norm_weights_layer(ell)

        ell = tf.divide(ell, 0.1)

        theta_weight = tf.nn.softmax(ell, axis=2) # shape (B,L,n_params_out)

        # sample from gumbel softmax parameterized by ell
        theta_sample = tf.argmax(tf.add(tfp.distributions.Gumbel(0, 1).sample(), ell), axis=2)  # shape (B, L)
        theta_sample = tf.one_hot(theta_sample, self.n_params_out)

        token_weight = tf.multiply(self.temperature, theta_weight) + tf.multiply(1-self.temperature, theta_sample) # shape (B, L, V)

        ###

        ell_amp = self.activation_amp_layer(self.dense_amp_layer(inputs))
        ell_amp = self.norm_amp_layer(ell_amp)

        ell_amp = tf.divide(ell_amp, 0.1)

        # sample from gumbel softmax parameterized by ell_amp
        theta_amp_sample = tf.argmax(tf.add(tfp.distributions.Gumbel(0, 1).sample(), ell_amp), axis=2)  # shape (B, L)
        token_amp = tf.one_hot(theta_amp_sample, self.n_params_amp_out) # shape (B, L, V)

        # use token_amp to look up the amplitude in self.amp_lookup
        token_amp = tf.gather(self.amp_lookup, tf.argmax(token_amp, axis=2)) # shape (B, L)
        # add a dimension to token_amp
        token_amp = tf.expand_dims(token_amp, axis=2) # shape (B, L, 1)

        token_weight = tf.multiply(token_weight, token_amp) # shape (B, L, V)

        return token_weight, self.temperature

##########

VOCAB_SIZE = 8
AMP_VOCAB_SIZE = VOCAB_SIZE
SEQ_LEN = 5

config = tokenize.Config(VOCAB_SIZE=VOCAB_SIZE,                        
                            LEARNING_RATE=0.0001,
                            RNN_UNITS=12)

inputs = tf.random.normal([1, SEQ_LEN, config.RNN_UNITS])

rnn_inference_layer = tf.keras.layers.GRU(
                                config.RNN_UNITS,
                                return_sequences=True,
                                stateful=False,
                                name='rnn_inference_layer'
                                )
rnn_inference_output = rnn_inference_layer(inputs)  # shape (B, L, RNN_UNITS)

token_weights_inference_layer = TokenWeightsAmpLayer(config.VOCAB_SIZE, AMP_VOCAB_SIZE, )  
token_weight, temperature = token_weights_inference_layer(rnn_inference_output)  # B, L, V

token_basis_layer = tf.keras.layers.Conv1D(
                        filters=config.VOCAB_SIZE,
                        kernel_size=(config.CONV_WIDTH,),
                        padding='same',
                        activation='linear',
                        strides=1,
                        name = 'token_basis_layer')

signal = token_basis_layer(token_weight)  # shape (B, L, V)
signal = tf.reduce_sum(signal, axis=2, keepdims=True) # shape (B, L, 1)

nll_layer = tokenize_layers.NegLogNormalLikelihoodLayer()
nll_loss = nll_layer([signal, tf.ones([1]), inputs])
