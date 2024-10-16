
import tensorflow as tf
import keras
from keras import layers
from dataclasses import dataclass
from dynamic_dict import DynamicDict



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


@dataclass
class GPTConfig:
    block_size: int = 512
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


class GPT:

    def __init__(self, config):

        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = DynamicDict(dict(
            wte = layers.Embedding(input_dim=config.vocab_size, output_dim=config.n_embd),
            wpe = layers.Embedding(input_dim=config.block_size, output_dim=config.n_embd),
            drop = layers.Dropout(config.dropout),
            h = [Block(config) for _ in range(config.n_layer)],
            ln_f = LayerNorm(config.n_embd, use_bias=config.bias),
            lm_head = layers.Dense(config.vocab_size, use_bias=False),
        ))

