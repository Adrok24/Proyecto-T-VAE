import tensorflow as tf
import pandas as pd


def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(x):
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(x)

    return tf.maximum(look_ahead_mask, padding_mask)


def tokenize_and_filter_df(inputs, tokenizer, max_length):
    tokenized_inputs = {}
    count_dict = 0
    for index, line in inputs.iterrows():
        sentence = line['line']
        sentence = tokenizer.encode(sentence)
        if len(sentence) <= max_length:
            tokenized_inputs[count_dict] = {"line": line['line'], 'secuence': sentence, "len": len(sentence), "genre": line['genre'], "sent": line['Sentimiento']}
            count_dict += 1
    data_filtered = pd.DataFrame.from_dict(tokenized_inputs, "index")

    return data_filtered


def tokenize_and_filter(inputs, tokenizer, max_length):
    tokenized_inputs = []
    for sentence1 in inputs:
        sentence1 = tokenizer.encode(sentence1)
        if len(sentence1) <= max_length:
            tokenized_inputs.append(sentence1)
  
    tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_inputs, maxlen=max_length, padding='post')

    return tokenized_inputs


def generate_from_normal(model, n_sentences=3, latent_space=32):
    errores = 0
    predictions = []
    for i in range(n_sentences):
        z_sampled = tf.random.normal(shape=(1,latent_space))
        prediction = model.decode_sample(z_sampled,silent_reconstruct=True)
        predictions.append(prediction)

    return predictions
  

