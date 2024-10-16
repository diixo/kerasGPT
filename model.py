
import tensorflow as tf
import keras
from keras import layers



""" layer: LayerNorm with optional bias"""
class LayerNorm(layers.Layer):

    def __init__(self, ndim, use_bias=True):
        super().__init__()
        self.weight = self.add_weight(
            shape=(ndim,), initializer="ones", trainable=True
        )
        if use_bias:
            self.bias = self.add_weight(
                shape=(ndim,), initializer="zeros", trainable=True
            )
        else:
            self.bias = None
        self.epsilon = 1e-5


    def call(self, inputs):
        mean = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        variance = tf.reduce_mean(tf.square(inputs - mean), axis=-1, keepdims=True)
        normalized_inputs = (inputs - mean) / tf.sqrt(variance + self.epsilon)

        output = self.weight * normalized_inputs
        if self.bias is not None:
            output += self.bias
        return output


class MLP(keras.Model):

    def __init__(self, n_embd, use_bias, dropout=0.1):
        super().__init__()
        self.fc = layers.Dense(4 * n_embd, use_bias=use_bias)
        self.gelu = layers.Activation(keras.activations.gelu)
        self.proj = layers.Dense(n_embd, use_bias=use_bias)
        self.dropout = layers.Dropout(dropout)

    def call(self, x):
        x = self.fc(x)
        x = self.gelu(x)
        x = self.proj(x)
        x = self.dropout(x)
        return x


"""
## Implement a Transformer block as a layer
"""
class Block(keras.Model):

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.ln_1 = LayerNorm(ndim=embed_dim, use_bias=True)
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=dropout)
        self.ln_2 = LayerNorm(ndim=embed_dim, use_bias=True)
        self.mlp = MLP(n_embd=embed_dim, use_bias=True, dropout=dropout) # ff_dim = 4*embed_dim

    def call(self, x):
        attn_output = self.attn(self.ln_1(x), self.ln_1(x))
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        return x


