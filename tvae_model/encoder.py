import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, Embedding, Reshape, Lambda
from . import MultiHeadAttention, PositionalEncoding, Sampling
from .utils import create_padding_mask


def encoder_layer(units, d_model, num_heads, dropout, name="encoder_layer"):
    inputs = Input(shape=(None, d_model), name="inputs")
    padding_mask = Input(shape=(1, 1, None), name="padding_mask")

    attention = MultiHeadAttention(
        d_model, num_heads, name="attention")({
            'query': inputs,
            'key': inputs,
            'value': inputs,
            'mask': padding_mask
        })
    attention = Dropout(rate=dropout)(attention)
    attention = LayerNormalization(
        epsilon=1e-6)(inputs + attention)

    outputs = Dense(units=units, activation='relu')(attention)
    outputs = Dense(units=d_model)(outputs)
    outputs = Dropout(rate=dropout)(outputs)
    outputs = LayerNormalization(epsilon=1e-6)(attention + outputs)

    return Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)


def encoder(vocab_size, num_layers, units, d_model, num_heads, dropout, latent_space, max_length, name="encoder"):
    inputs = Input(shape=(None,), name="inputs")
    padding_mask = Lambda(
        create_padding_mask, output_shape=(1, 1, None),
        name='enc_padding_mask')(inputs)

    embeddings = Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
    outputs = Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):
        outputs = encoder_layer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name="encoder_layer_{}".format(i),
        )([outputs, padding_mask])

    outputs = Reshape([max_length * d_model])(outputs)
    outputs = Dense(max_length * 10)(outputs)
    outputs = Dense(latent_space)(outputs)

    mu = Dense(latent_space, name='mu')(outputs)  # TODO: revisar la dimensionalidad del espacio latente

    logvar = Dense(latent_space, name='logvar')(outputs)
    z = Sampling(name='encoder_output')([mu, logvar])

    return Model(inputs=inputs, outputs=[mu, logvar, z], name=name)
