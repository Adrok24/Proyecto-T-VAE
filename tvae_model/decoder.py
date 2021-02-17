from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, TimeDistributed, LSTM, Reshape
from . import MultiHeadAttention


def decoder_layer(units, d_model, num_heads, dropout, name="decoder_layer"):
    decoder_input = Input(shape=(None, d_model), name="decoder_input_layer")
    attention1 = MultiHeadAttention(
        d_model, num_heads, name="attention_1")(inputs={
            'query': decoder_input,
            'key': decoder_input,
            'value': decoder_input,
            'mask': None
        })

    attention1 = LayerNormalization(
        epsilon=1e-6)(attention1 + decoder_input)

    outputs = Dense(units=units, activation='relu')(attention1)
    outputs = Dense(units=d_model)(outputs)
    outputs = Dropout(rate=dropout)(outputs)

    outputs = LayerNormalization(epsilon=1e-6)(outputs + attention1)

    return Model(inputs=[decoder_input], outputs=outputs, name=name)


def decoder(vocab_size, num_layers, units, d_model, num_heads,
            dropout, latent_space, max_length, name='decoder'):
    decoder_input = Input(shape=latent_space, name='decoder_input')
    output = Dense(max_length * 10)(decoder_input)
    output = Dense(max_length * d_model, name='linear_proyection')(output)
    output = Reshape([max_length, d_model])(output)

    outputs = Dropout(rate=dropout)(output)

    for i in range(num_layers):
        outputs = decoder_layer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name='decoder_layer_{}'.format(i),
        )(inputs=[outputs])

    outputs = TimeDistributed(Dense(d_model))(outputs)
    outputs = LSTM(units=d_model, return_sequences=True, name='LSTM')(outputs)
    outputs = TimeDistributed(Dense(d_model))(outputs)
    outputs = Dense(units=vocab_size, name="outputs")(outputs)

    return Model(inputs=[decoder_input], outputs=outputs, name=name)
