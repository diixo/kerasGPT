
import math
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
                shape=(ndim,), initializer="zeros", trainable=True)
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
    bias: bool = True   # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


class GPT:

    def __init__(self, config):

        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = DynamicDict(dict(
            wte = layers.Embedding(input_dim=config.vocab_size, output_dim=config.n_embd),
            wpe = layers.Embedding(input_dim=config.block_size, output_dim=config.n_embd),
            drop = layers.Dropout(config.dropout),
            h = [ Block(config)
                    for _ in range(config.n_layer)
                ],
            ln_f = LayerNorm(config.n_embd, use_bias=config.bias),
            lm_head = layers.Dense(config.vocab_size, use_bias=False),
        ))

        self.transformer.wte.embeddings = self.lm_head.kernel

        self._init_weights()


    def _init_weights(self):
        for layer in self.layers:
            if isinstance(layer, layers.Dense):
                std = 0.02 / math.sqrt(2 * self.config.n_layer)
                layer.kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=std)
            elif isinstance(layer, layers.Embedding):
                layer.embeddings_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)


    def forward(self, idx, targets=None):

        b, t = tf.shape(idx)[0], tf.shape(idx)[1]
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        
        pos = tf.range(0, t, dtype=tf.int32)    # shape (t)

        tok_emb = self.transformer.wte(idx)     # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)     # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = tf.keras.losses.sparse_categorical_crossentropy(targets, logits, from_logits=True)
        else:
            logits = self.lm_head(x[:, -1, :])
            loss = None

        return logits, loss
