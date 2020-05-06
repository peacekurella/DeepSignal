import tensorflow as tf
from .SequenceEncoder import SequenceEncoder


class DifferenceEncoder(tf.keras.Model):

    def __init__(self, enc_units, batch_size, num_layers, dropout_rate):
        """
        constructor for Transformation Encoder class
        :param enc_units: SequenceEncoder Hidden Units
        :param batch_size: batch_size
        :param num_layers: number of layers in the sequence encoder
        :param dropout_rate: Encoder Dropout rate
        """

        # call the super constructor
        super(DifferenceEncoder, self).__init__()

        self.encoderA = SequenceEncoder(
            enc_units,
            batch_size,
            num_layers,
            dropout_rate
        )

        self.encoderB = SequenceEncoder(
            enc_units,
            batch_size,
            num_layers,
            dropout_rate
        )

    def call(self, inputs, zero_state, training):
        """
        Foward pass through the network
        :param inputs: Tuple of input sequences to each encoder
        :param training: Train mode
        :return: output, state
        """

        # encoderA output
        encA_output, encA_hidden = self.encoderA(inputs[0], zero_state, training)

        # encoderB output
        encB_output, encB_hidden = self.encoderB(inputs[1], zero_state, training)

        return (encA_output - encB_output), (encA_hidden - encB_hidden)
