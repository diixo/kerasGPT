
import math
import tensorflow as tf
import keras
import keras.backend as K
from keras import layers
from dataclasses import dataclass
from dynamic_dict import DynamicDict



""" 
## layer: LayerNorm with optional bias
"""
class LayerNorm():

    def __init__(self, ndim, use_bias=True):
        self.weight = self.add_weight(
            shape=(ndim,), initializer="ones", trainable=True
        )
        if use_bias:
            self.bias = self.add_weight(
                shape=(ndim,), initializer="zeros", trainable=True)
        else:
            self.bias = None
        self.epsilon = 1e-5


    # training flag is don't care
    def call(self, inputs, training=False):
        mean = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        variance = tf.reduce_mean(tf.square(inputs - mean), axis=-1, keepdims=True)
        normalized_inputs = (inputs - mean) / tf.sqrt(variance + self.epsilon)

        output = self.weight * normalized_inputs
        if self.bias is not None:
            output += self.bias
        return output


class MLP():

    def __init__(self, n_embd, use_bias, dropout=0.1):
        self.fc = layers.Dense(4 * n_embd, use_bias=use_bias)
        self.gelu = layers.Activation(keras.activations.gelu)
        self.proj = layers.Dense(n_embd, use_bias=use_bias)
        self.dropout = layers.Dropout(dropout)


    def call(self, x, training=False):
        x = self.fc(x)
        x = self.gelu(x)
        x = self.proj(x)
        x = self.dropout(x, training=training)
        return x

    
    # special scaled init to the residual projections, per GPT-2 paper
    def init_weights(self):
        if isinstance(self.fc, keras.layers.Dense):
            initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
            self.fc.build((None, self.fc.units))  # used 4 * n_embd
            self.fc.kernel.assign(initializer(self.fc.kernel.shape))
            
            if self.fc.bias is not None:
                bias_initializer = keras.initializers.Zeros()
                self.fc.bias.assign(bias_initializer(self.fc.bias.shape))
        
        if isinstance(self.proj, keras.layers.Dense):
            initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
            self.proj.build((None, self.proj.units))  # used n_embd
            self.proj.kernel.assign(initializer(self.proj.kernel.shape))
            
            if self.proj.bias is not None:
                bias_initializer = keras.initializers.Zeros()
                self.proj.bias.assign(bias_initializer(self.proj.bias.shape))



"""
## Implement TransformerBlock
"""
class Block():

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        self.ln_1 = LayerNorm(ndim=embed_dim, use_bias=True)
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=dropout)
        self.ln_2 = LayerNorm(ndim=embed_dim, use_bias=True)
        self.mlp = MLP(n_embd=embed_dim, use_bias=True, dropout=dropout) # ff_dim = 4*embed_dim


    def call(self, x, training=False):
        attn_output = self.attn(self.ln_1(x), self.ln_1(x), training=training)
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x), training)
        return x
    
    def init_weights(self):
        self.mlp.init_weights()


@dataclass
class GPTConfig:
    block_size: int = 512
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 (and 48) for efficiency
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = True   # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


"""
## Build the GPT model class as a `keras.Model` subclass
"""
class GPTModel(keras.Model):

    def __init__(self, config):
        super().__init__()
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
        # force initialization
        for block in self.transformer.h: block.init_weight()


    def call(self, idx, targets=None):

        b, t = tf.shape(idx)[0], tf.shape(idx)[1] # shape(batch_size, sequence_length)
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


    """
    Generate indexed based sequence : idx (Tensor, shape (b, t)).
    """
    @tf.function # create function as executive graph
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.shape[1] <= self.block_size else idx[:, -self.block_size:]

            # use without gradients
            with tf.no_grad():
                logits, _ = self(idx_cond)

            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = tf.math.top_k(logits, top_k)
                logits = tf.where(logits < v[:, -1:], -float('Inf'), logits)

            probs = tf.nn.softmax(logits, axis=-1)

            if do_sample:
                idx_next = tf.random.categorical(probs, num_samples=1)
            else:
                _, idx_next = tf.math.top_k(probs, k=1, sorted=False)

            idx = tf.concat([idx, idx_next], axis=1)

        return idx


    def get_num_embeddings(self):
        return round(1 / 1000) / 1000
