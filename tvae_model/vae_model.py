import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras.models import Model
from contextlib import suppress
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, Embedding,\
    TimeDistributed, LSTM, Reshape, Lambda
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from . import MultiHeadAttention, PositionalEncoding, Sampling
from .utils import create_padding_mask, tokenize_and_filter


beta = K.variable(value=0.0)
beta._trainable = False


class VAEModel(Model):
    def __init__(self, a, b, vocab_size, num_layers, units, d_model, num_heads, dropout, latent_space,
                 tokenizer, max_length=32, mask=None, function=None, **kwargs):
        super(VAEModel, self).__init__(**kwargs)
        self.a = a
        self.b = b
        self.function = function
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.num_layers = num_layers
        self.units = units
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.latent_space = latent_space
        self.vocab_size = vocab_size
        self.mask = mask

    def encoder_layer(self, name="encoder_layer"):
        inputs = Input(shape=(None, self.d_model), name="inputs")
        padding_mask = Input(shape=(1, 1, None), name="padding_mask")

        attention = MultiHeadAttention(
            self.d_model, self.num_heads, name="attention")({
                'query': inputs,
                'key': inputs,
                'value': inputs,
                'mask': padding_mask
            })
        attention = Dropout(rate=self.dropout)(attention)
        attention = LayerNormalization(
            epsilon=1e-6)(inputs + attention)

        outputs = Dense(units=self.units, activation='relu')(attention)
        outputs = Dense(units=self.d_model)(outputs)
        outputs = Dropout(rate=self.dropout)(outputs)
        outputs = LayerNormalization(epsilon=1e-6)(attention + outputs)

        return Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)

    def encoder(self, name="encoder"):
        inputs = Input(shape=(None,), name="inputs")
        padding_mask = Lambda(
            create_padding_mask, output_shape=(1, 1, None),
            name='enc_padding_mask')(inputs)

        embeddings = Embedding(self.vocab_size, self.d_model)(inputs)
        embeddings *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        embeddings = PositionalEncoding(self.vocab_size, self.d_model)(embeddings)
        outputs = Dropout(rate=self.dropout)(embeddings)

        for i in range(self.num_layers):
            outputs = self.encoder_layer(
                name="encoder_layer_{}".format(i),
            )([outputs, padding_mask])

        outputs = Reshape([self.max_length * self.d_model])(outputs)
        outputs = Dense(self.max_length * 10)(outputs)
        outputs = Dense(self.latent_space)(outputs)

        mu = Dense(self.latent_space, name='mu')(outputs)  # TODO: revisar la dimensionalidad del espacio latente

        logvar = Dense(self.latent_space, name='logvar')(outputs)
        z = Sampling(name='encoder_output')([mu, logvar])

        return Model(inputs=inputs, outputs=[mu, logvar, z], name=name)

    def decoder_layer(self, name="decoder_layer"):
        decoder_input = Input(shape=(None, self.d_model), name="decoder_input_layer")
        attention1 = MultiHeadAttention(
            self.d_model, self.num_heads, name="attention_1")(inputs={
                'query': decoder_input,
                'key': decoder_input,
                'value': decoder_input,
                'mask': None
            })

        attention1 = LayerNormalization(
            epsilon=1e-6)(attention1 + decoder_input)

        outputs = Dense(units=self.units, activation='relu')(attention1)
        outputs = Dense(units=self.d_model)(outputs)
        outputs = Dropout(rate=self.dropout)(outputs)

        outputs = LayerNormalization(epsilon=1e-6)(outputs + attention1)

        return Model(inputs=[decoder_input], outputs=outputs, name=name)

    def decoder(self, name='decoder'):
        decoder_input = Input(shape=self.latent_space, name='decoder_input')
        output = Dense(self.max_length * 10)(decoder_input)
        output = Dense(self.max_length * self.d_model, name='linear_proyection')(output)
        output = Reshape([self.max_length, self.d_model])(output)

        outputs = Dropout(rate=self.dropout)(output)

        for i in range(self.num_layers):
            outputs = self.decoder_layer(
                name='decoder_layer_{}'.format(i),
            )(inputs=[outputs])

        outputs = TimeDistributed(Dense(self.d_model))(outputs)
        outputs = LSTM(units=self.d_model, return_sequences=True, name='LSTM')(outputs)
        outputs = Dense(units=self.vocab_size, name="outputs")(outputs)

        return Model(inputs=[decoder_input], outputs=outputs, name=name)

    def loss_function(self, y_true, y_pred):
        print("ytrue:", y_true, "ypred:", y_pred)
        y_true = tf.reshape(y_true, shape=(-1, self.max_length))
      
        loss = SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')(y_true, y_pred)

        return tf.reduce_mean(loss)

    def train_step(self, data):
        data = data[0]
        with tf.GradientTape() as tape:            
            
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            reconstruction_loss = self.loss_function(data, reconstruction)

            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_sum(kl_loss, axis=1)
            kl_loss *= -0.5
            if self.function == 'linear':
                factor = self.a * beta + self.b
            else:
                factor = 1 / (1 + tf.exp(-self.a * beta + self.b))

            total_loss = reconstruction_loss + kl_loss * factor

        grads = tape.gradient(total_loss, self.trainable_weights)
        
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "kl_weight": factor,
        }

    def reconstruct(self, q_sample, silent_reconstruct=False, silent_orig=True):
        _, _, result = self.encoder(q_sample.reshape(1, self.max_length, 1))
        print('Reconstr: ', self.decode_sample(result, silent_reconstruct))
        print('Original: ', self.tokenizer.decode([i for i in q_sample if i < self.tokenizer.vocab_size]))
        if not silent_orig:
            print(q_sample)

    def decode_sample(self, z_sampled, silent_reconstruct=False):
        vmodel_decoder = self.decoder()
        result = vmodel_decoder(z_sampled)
        with suppress(Exception):
            prediction = np.argmax(result, axis=2)

        pred = prediction[0]
        pred_to_decode = [i for i in pred if i < self.tokenizer.vocab_size]
        if not silent_reconstruct:
            print(pred_to_decode)
        return self.tokenizer.decode(pred_to_decode)

    def reconstruct_sentence(self, sentence, silent_reconstruct=False, silent_orig=True):
        sentence = [sentence]
        sentence = tokenize_and_filter(sentence, self.tokenizer, self.max_length)
        sentence = sentence.reshape(self.max_length)
        return self.reconstruct(sentence, silent_reconstruct, silent_orig)

    def call(self, inputs):
        vmodel_encoder = self.encoder()
        vmodel_decoder = self.decoder()
        mu, logvar, z = vmodel_encoder(inputs)
        print("checkpoint_call")
        return vmodel_decoder(z)
