import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense


def scaled_dot_product_attention(query, key, value, mask):
    """Calculamos attention weights """

    matmul_qk = tf.matmul(query, key, transpose_b=True)
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    if mask is not None:
        logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(logits, axis=-1)
    output = tf.matmul(attention_weights, value)

    return output


class MultiHeadAttention(tf.keras.layers.Layer):
    ''' Vamos a armar la capa para multihead attention '''

    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)

        ## que es este super?
        ## super() te permite acceeder a los metodos de la super clase de la cual
        ## la subclase está heredando. En este caso, estas herendando de Layers.

        ## definimos algunos parametros: cuantas cabezas va a tener self attention
        ## y la dimensionalidad del embedding
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        ## cuantas dimensiones va a tener cada cabeza:
        self.depth = d_model // self.num_heads

        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)

        self.dense = tf.keras.layers.Dense(units=d_model)

    def split_heads(self, inputs, batch_size):
        ## vamos a armar la division en cabezas.
        ## se va a entender mejor en el siguiente bloque de codigo
        ## por ahora es solamente la forma en la que
        ## reacomodamos los datos para armar las cabezas
        inputs = tf.reshape(
            inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_heads': self.num_heads,
            'd_model': self.d_model,
            'depth': self.depth,
            'query_dense': self.query_dense,
            'key_dense': self.key_dense,
            'value_dense': self.value_dense,
            'dense': self.dense
        })
        return config

    def call(self, inputs):
        ''' Este call es el metodo que  va a llamar keras para usar la capa'''

        query, key, value, mask = inputs['query'], inputs['key'], inputs[
            'value'], inputs['mask']
        batch_size = tf.shape(query)[0]

        # linear layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # print(self.name, mask)

        # split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # scaled dot-product attention
        scaled_attention = scaled_dot_product_attention(query, key, value, mask)

        # acomodamos las dimensiones
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # concatenamos las cabezas
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))

        # final linear layer
        outputs = self.dense(concat_attention)

        return outputs