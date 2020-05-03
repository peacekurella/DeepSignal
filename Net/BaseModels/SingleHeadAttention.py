import tensorflow as tf


class SingleHeadAttention(tf.keras.layers.Layer):
    """
    courtesy of TensorFlow tutorials:
    https://www.tensorflow.org/tutorials/text/nmt_with_attention
    """

    def __init__(self, units):
        """
        Constructor for Attention class
        :param units: number of hidden units for attention calculation
        """
        super(SingleHeadAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        """
        Defines forward pass operation :param query: hidden state from decoder of shape (batch_size, enc_units)
        :param query: Hidden state of the decoder
        :param values: encoder output tensor of shape (batch_size, sequence_length, input_size) :return: context
        vector, attention weights of shapes (batch size, enc_units) and (batch_size, sequence_length, 1)
        """

        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # we are doing this to broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights
