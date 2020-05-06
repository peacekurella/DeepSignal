import tensorflow as tf
from .SingleHeadAttention import SingleHeadAttention


class SequenceDecoder(tf.keras.Model):

    def __init__(self, output_size, dec_units, batch_size, num_layers, dropout_rate, enable_batch_norm):
        """
        Constructor for the decoder class
        :param enable_batch_norm:
        :param output_size: The size of the output vector
        :param dec_units: Number of hidden units in the GRU cell
        :param batch_size: batch size
        :param num_layers: number of layers in the encoder
        :param dropout_rate: dropout rate for regularization
        """

        # call the super constructor
        super(SequenceDecoder, self).__init__()

        # initialize hidden units in decoder
        self.dec_units = dec_units

        # initialize batch size
        self.batch_size = batch_size

        # initialize number of layers
        self.num_layers = num_layers

        # initialize output size
        self.output_size = output_size

        # initialize GRU layers
        self.GRU = []
        for _ in range(num_layers):
            self.GRU.append(
                tf.keras.layers.GRU(
                    self.dec_units,
                    return_sequences=True,
                    return_state=True
                )
            )

        # initialize finGru
        self.finGRU = tf.keras.layers.GRU(
            self.dec_units,
            return_sequences=True,
            return_state=True
        )

        # initialize output dense layer
        self.Fc = tf.keras.layers.Dense(
            output_size
        )

        # initialize hidden state dense layers
        self.hidFc = []
        for _ in range(num_layers):
            self.hidFc.append(
                tf.keras.layers.Dense(
                    dec_units,
                    activation='relu'
                )
            )

        # initialize attention layer
        self.Attention = SingleHeadAttention(self.dec_units)

        # initialize drop out layer
        self.Dropout = tf.keras.layers.Dropout(
            rate=dropout_rate
        )

        # initialize layer norm
        self.layerNorm = tf.keras.layers.LayerNormalization(
            center=False,
            scale=False
        )

        # batch_norm flag
        self.enable_batch_norm = enable_batch_norm

        # initialize batchNorm
        self.batchNorm = []
        for _ in range(num_layers):
            self.batchNorm.append(
                tf.keras.layers.BatchNormalization(
                    center=False,
                    scale=False
                )
            )

    def call(self, prev_output, hidden, enc_output, train):
        """
        Defines forward pass operation
        :param prev_output: previous output tensor of shape (batch_size, input_shape)
        :param hidden: hidden state tensor of shape (num_layers, batch_size, enc_units)
        :param enc_output: encoder output tensor of shape (batch_size, seqLengtf.reshape(hidden[0], (self.batch_size, self.dec_units))th, enc_units)
        :param train: boolean value indicating training mode operation
        :return: output, state, and attention_weight tensors of shapes
                (batch_size, output_size), (num_layers, batch_size, dec_units), and (batch_size, seqLength, 1)
        """

        states = []

        # get attention weights and context vector
        input_hidden = hidden[self.num_layers - 1]
        context_vector, attention_weights = self.Attention(input_hidden, enc_output)

        for i in range(self.num_layers):

            # create initial state
            if hidden[i].shape[1] == self.dec_units:
                initial_state = hidden[i]
            else:
                initial_state = self.hidFc[i](hidden[i])

            # add dropout and batchNorm
            if self.enable_batch_norm:
                prev_output = self.batchNorm[i](prev_output)
            prev_output = self.Dropout(prev_output)

            # reshape for GRU input
            prev_output = tf.expand_dims(prev_output, axis=1)

            # pass through GRU
            prev_output, state = self.GRU[i](prev_output, initial_state=initial_state)
            prev_output = tf.squeeze(prev_output, axis=1)
            states.append(tf.expand_dims(state, axis=0))

        # concatenate context vector with prev_output
        prev_output = tf.expand_dims(prev_output, axis=1)
        prev_output = tf.concat([tf.expand_dims(context_vector, 1), prev_output], axis=-1)

        # pass through final GRU layer
        output, state = self.finGRU(prev_output)

        # reshape output
        output = tf.reshape(output, (-1, output.shape[2]))

        # add a dense layer for output
        output = self.Fc(output)
        state = tf.concat(states, axis=0)

        return output, state, attention_weights
