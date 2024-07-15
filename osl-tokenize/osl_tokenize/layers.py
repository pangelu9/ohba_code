'''

Layers

'''

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_probability as tfp
tfb = tfp.bijectors

class TokenWeightsLayer(layers.Layer):

    '''
    Layer to calculate the token weights for the OSL model
    
    Parameters
    ----------
    n_params_out : int
        Number of parameters to output
    activation : str
        Activation function for the dense layer
    name : str
        Name of the layer
    
    '''

    def __init__(self, n_params_out, activation="linear", name=None, **kwargs):
        super().__init__(**kwargs)

        self.n_params_out = n_params_out
        self.dense_layer = layers.Dense(n_params_out,
                                 activation="linear",
                                 name=name)

        self.temperature = tf.Variable(tf.ones([1,]), trainable=False)
        self.norm_layer = layers.LayerNormalization(center=True, scale=True)
        self.activation_layer = layers.Activation(activation)

    def call(self, inputs, **kwargs):
        ell = self.activation_layer(self.dense_layer(inputs))
        ell = self.norm_layer(ell)

        ell = tf.divide(ell, 0.1)

        theta_weight = tf.nn.softmax(ell, axis=2) # shape (B,L,n_params_out)

        # sample from gumbel softmax parameterized by ell
        theta_sample = tf.argmax(tf.add(tfp.distributions.Gumbel(0, 1).sample(), ell), axis=2)  # shape (B, L)
        theta_sample = tf.one_hot(theta_sample, self.n_params_out)

        token_weight = tf.multiply(self.temperature, theta_weight) + tf.multiply(1-self.temperature, theta_sample)

        return token_weight, self.temperature

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

class NegLogNormalLikelihoodLayer(layers.Layer):

    '''
    Negative log-likelihood layer
    '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def call(self, inputs):
                
        signal, noise, data = inputs

        norm = tfp.distributions.Normal(loc=signal, scale=noise)

        ll_loss = norm.log_prob(data)  # B, L

        # Sum over time dimension
        ll_loss = tf.reduce_sum(ll_loss, axis=1) # B

        # Average over the batch dimension
        ll_loss = tf.reduce_mean(ll_loss, axis=0)

        # Add the negative log-likelihood to the loss
        nll_loss = -ll_loss
        self.add_loss(nll_loss)

        return tf.expand_dims(nll_loss, axis=-1)

