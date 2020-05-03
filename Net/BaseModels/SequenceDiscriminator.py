import tensorflow as tf
from .SequenceEncoder import SequenceEncoder


class SequenceDiscriminator(tf.keras.Model):

    def __init__(self, enc_units, batch_size, dropout_rate):
        """
        Initializes the discriminator
        :param enc_units: Number of hidden units in the LSTM cell
        :param batch_size: batch size
        :param dropout_rate: Dropout rate in the encoder
        """

        # call the super constructor
        super(SequenceDiscriminator, self).__init__()

        # initialize the input encoder
        self.input_enc = SequenceEncoder(enc_units, batch_size, 1, dropout_rate)

        # initialize the dense layer
        self.fc = tf.keras.layers.Dense(1)

        # initialize batch_size
        self.batch_size = batch_size

    def call(self, input_seq):
        """
        Defines a forward pass through the discriminator
        :param input_seq: Input sequence to the discriminator (batch_size, time_steps, input_shape)
        :param target_seq: target sequence of the discriminator  (batch_size, time_steps, input_shape)
        :return: Binary output of the discriminator (batch_size, 1)
        """

        # initialize the zero state
        zero_state = self.input_enc.initialize_hidden_state(self.batch_size)

        # encode the input sequence
        _, input_state = self.input_enc(input_seq, zero_state, True)

        # calculate the transformed state
        transformed = input_state[0]

        # classify the transformation as real or fake
        output = self.fc(transformed)

        return output
