import tensorflow as tf
from .SequenceEncoder import SequenceEncoder

class TransformationEncoder(tf.keras.Model):

    def __init__(self, enc_units, batch_size, num_layers, dropout_rate):
        """
        constructor for Tranformation Encoder class
        :param enc_units: SequenceEncoder Hidden Units
        :param batch_size: batch_size
        :param num_layers: number of layers in the sequence encoder
        :param dropout_rate: Encoder Dropout rate
        """

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

    def call(self, inputs, training):
        """
        Foward pass through the network
        :param inputs: Tuple of input sequences to each encoder
        :param training: Train mode
        :return: output, state
        """

        # initialize encoder hidden state
        zero_state = self.encoderA.initialize_hidden_state(self.encoderA.batch_size)

        # encoderA output
        encA_output, encA_hidden = self.encoderA(inputs[0], zero_state, training)

        # encoderB output
        encB_output, encB_hidden = self.encoderB(inputs[1], zero_state, training)

        return (encA_output - encB_output), (encA_hidden - encB_hidden)