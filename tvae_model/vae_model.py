import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras.models import Model
from contextlib import suppress
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from .utils import tokenize_and_filter


beta = K.variable(value=0.0)
beta._trainable = False


class VAEModel(Model):
    def __init__(self, encoder, decoder, a, b, vocab_size, num_layers, units, d_model, num_heads, dropout, latent_space,
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
        self.encoder = encoder
        self.decoder = decoder

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
        result = self.decoder(z_sampled)
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
        mu, logvar, z = self.encoder(inputs)
        print("checkpoint_call")
        return self.decoder(z)
