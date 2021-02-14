import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, LayerNormalization, TimeDistributed, LSTM
import pandas as pd


def create_padding_mask(x):
  mask = tf.cast(tf.math.equal(x, 0), tf.float32)

  return mask[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(x):
  seq_len = tf.shape(x)[1]
  look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
  padding_mask = create_padding_mask(x)

  return tf.maximum(look_ahead_mask, padding_mask)

def tokenize_and_filter_df(inputs):
  tokenized_inputs = {}
  count_dict = 0
  for index, line in inputs.iterrows():
    sentence = line['line']
    sentence = tokenizer.encode(sentence)
    # filtramos
    if len(sentence) <= MAX_LENGTH:
      tokenized_inputs[count_dict] = {"line": sentence, "len": len(sentence), "genre": line['genre']}
      count_dict += 1

  data_filtered = pd.DataFrame.from_dict(tokenized_inputs, "index")
  
  return data_filtered

def tokenize_and_filter(inputs):
  tokenized_inputs = []
  
  for sentence1 in inputs:
    sentence1 = tokenizer.encode(sentence1)
    if len(sentence1) <= MAX_LENGTH:
      tokenized_inputs.append(sentence1)
  
  tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
      tokenized_inputs, maxlen=MAX_LENGTH, padding='post')
  
  return tokenized_inputs
