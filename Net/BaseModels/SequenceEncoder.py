import tensorflow as tf


class SequenceEncoder(tf.keras.Model):

    def __init__(self, enc_units, batch_size, num_layers, dropout_rate):
        """
        Constructor for Encoder class
        :param enc_units: Number of hidden units in the GRU cell
        :param batch_size: batch size
        :param num_layers: number of layers in the encoder
        :param dropout_rate: dropout rate for regularization
        """

        # call the super constructor
        super(SequenceEncoder, self).__init__()

        # define batch size
        self.batch_size = batch_size

        # initialize GRU hidden state size
        self.enc_units = enc_units

        # initialize number of layers
        self.num_layers = num_layers

        # initialize GRU layers
        self.GRU = []
        for _ in range(num_layers):
            self.GRU.append(
                tf.keras.layers.GRU(
                    self.enc_units,
                    return_sequences=True,
                    return_state=True
                )
            )

        # initialize drop out layer
        self.Dropout = tf.keras.layers.Dropout(
            rate=dropout_rate
        )

        # initialize layer norm
        self.layerNorm = tf.keras.layers.LayerNormalization(
            center=False,
            scale=False
        )

        # initialize dense layers
        self.Fc = []
        for _ in range(num_layers):
            self.Fc.append(
                tf.keras.layers.Dense(
                    enc_units,
                    activation='relu'
                )
            )

    def call(self, x, hidden, train):
        """
        Defines forward pass operation :param x: input tensor to GRU cell of shape (batch_size, sequence_length,
        input_size) :param hidden: input hidden state of shape (layers, batch_size, enc_units) to GRU cell :param
        train: boolean variable indicating if it's train or inference :return: output, state tensors of shapes (
        batch_size, sequence_length, input_size) and (layers, batch_size, enc_units)
        """

        # initialize states
        states = []

        # create initial state
        initial_state = hidden[0]

        # add dropout and layerNorm
        x = self.layerNorm(x)
        x = self.Dropout(x)

        # pass through GRU
        output, state = self.GRU[0](x, initial_state=initial_state)
        states.append(tf.expand_dims(state, axis=0))

        # add additional layers as required
        for i in range(self.num_layers - 1):
            # create initial state
            initial_state = tf.reshape(hidden[i + 1], (self.batch_size, self.enc_units))

            # pass input through dense layer for the right shape
            dense_x = self.Fc[i](x)

            # add a short circuit for gradient flow
            output = tf.add(output, dense_x)

            # add dropout and layerNorm
            output = self.layerNorm(output)
            output = self.Dropout(output)

            # pass through GRU
            output, state = self.GRU[i + 1](output, initial_state=initial_state)
            states.append(tf.expand_dims(state, axis=0))

        # concatenate all hidden states
        state = tf.concat(states, axis=0)

        # add layer normalization
        output = self.layerNorm(output)

        return output, state

    def initialize_hidden_state(self, batch_size):
        """
        Returns a zero hidden state
        :return: a tensor of size (batch_size, enc_units)
        """
        return tf.zeros((self.num_layers, batch_size, self.enc_units))
