import tensorflow as tf

# configuration changes for RTX enabled devices
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


class Encoder(tf.keras.Model):

    def __init__(self, enc_units, batch_size, num_layers, dropout_rate):
        """
        Constructor for Encoder class
        :param enc_units: Number of hidden units in the GRU cell
        :param batch_size: batch size
        :param num_layers: number of layers in the encoder
        :param dropout_rate: dropout rate for regularization
        """

        # call the super constructor
        super(Encoder, self).__init__()

        # define batch size
        self.batch_size = batch_size

        # initialize GRU hidden state size
        self.enc_units = enc_units

        # initialize number of layers
        self.num_layers = num_layers

        # initialize GRU layer
        self.GRU = tf.keras.layers.GRU(
            self.enc_units,
            return_sequences=True,
            return_state=True
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

        # initialize dense layer
        self.Fc = tf.keras.layers.Dense(
            self.enc_units
        )

    def call(self, x, hidden, train):
        """
        Defines forward pass operation
        :param x: input tensor to GRU cell of shape (batch_size, sequence_length, input_size)
        :param hidden: input hidden state of shape (layers, batch_size, enc_units) to GRU cell
        :param train: boolean variable indicating if it's train or inference
        :return: output, state tensors of shapes (batch_size, sequence_length, input_size) and (layers, batch_size, enc_units)
        """

        # initialize states
        states = []

        # create initial state
        initial_state = tf.reshape(hidden[0], (self.batch_size, self.enc_units))

        # add dropout and layerNorm
        x = self.layerNorm(x)
        x = self.Dropout(x)

        # pass through GRU
        output, state = self.GRU(x, initial_state=initial_state)
        states.append(tf.reshape(state, (1, self.batch_size, self.enc_units)))

        # add additional layers as required
        for i in range(self.num_layers - 1):
            # create initial state
            initial_state = tf.reshape(hidden[i + 1], (self.batch_size, self.enc_units))

            # pass input through dense layer for the right shape
            dense_x = self.Fc(x)

            # add a short circuit for gradient flow
            output = tf.add(output, dense_x)

            # add dropout and layerNorm
            output = self.layerNorm(output)
            output = self.Dropout(output)

            # pass through GRU
            output, state = self.GRU(output, initial_state=initial_state)
            states.append(tf.reshape(state, (1, self.batch_size, self.enc_units)))

        # concatenate all hidden states
        state = tf.concat(states, axis=0)

        # add layer normalization
        output = self.layerNorm(output)

        return output, state

    def initialize_hidden_state(self):
        """
        Returns a zero hidden state
        :return: a tensor of size (batch_size, enc_units)
        """
        return tf.zeros((self.num_layers, self.batch_size, self.enc_units))


class Attention(tf.keras.layers.Layer):
    """
    courtesy of TensorFlow tutorials:
    https://www.tensorflow.org/tutorials/text/nmt_with_attention
    """

    def __init__(self, units):
        """
        Constructor for Attention class
        :param units: number of hidden units for attention calculation
        """
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        """
        Defines forward pass operation
        :param query: hidden state from decoder of shape (batch_size, enc_units)
        :param values: encoder output tensor of shape (batch_size, sequence_length, input_size)
        :return: context vector, attention weights of shapes (batch size, enc_units) and (batch_size, sequence_length, 1)
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


class Decoder(tf.keras.Model):

    def __init__(self, output_size, dec_units, batch_size, num_layers, dropout_rate):
        """
        Constructor for the decoder class
        :param output_size: The size of the output vector
        :param dec_units: Number of hidden units in the GRU cell
        :param batch_size: batch size
        :param num_layers: number of layers in the encoder
        :param dropout_rate: dropout rate for regularization
        """

        # call the super constructor
        super(Decoder, self).__init__()

        # initialize hidden units in decoder
        self.dec_units = dec_units

        # initialize batch size
        self.batch_size = batch_size

        # initialize number of layers
        self.num_layers = num_layers

        # initialize output size
        self.output_size = output_size

        # initialize GRU layer
        self.GRU = tf.keras.layers.GRU(
            self.dec_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )

        # initialize output dense layer
        self.Fc = tf.keras.layers.Dense(
            output_size
        )

        # initialize output dense layer
        self.hidFc = tf.keras.layers.Dense(
            dec_units,
            activation='relu'
        )
        # initialize attention layer
        self.Attention = Attention(self.dec_units)

        # initialize drop out layer
        self.Dropout = tf.keras.layers.Dropout(
            rate=dropout_rate
        )

        # initialize layer norm
        self.layerNorm = tf.keras.layers.LayerNormalization(
            center=False,
            scale=False
        )

        # initialize batchNorm
        self.batchNorm = tf.keras.layers.BatchNormalization(
            center=False,
            scale=False
        )

    def call(self, prev_output, hidden, enc_output, train):
        """
        Defines forward pass operation
        :param prev_output: previous output tensor of shape (batch_size, input_shape)
        :param hidden: hidden state tensor of shape (num_layers, batch_size, enc_units)
        :param enc_output: encoder output tensor of shape (batch_size, seqLength, enc_units)
        :param train: boolean value indicating training mode operation
        :return: output, state, and attention_weight tensors of shapes
                (batch_size, output_size), (batch_size, seqLength, enc_units), and (batch_size, seqLength, 1)
        """

        # get attention weights and context vector
        input_hidden = hidden[self.num_layers - 1]
        context_vector, attention_weights = self.Attention(input_hidden, enc_output)

        # create initial state
        if hidden[0].shape[1] == self.dec_units:
            initial_state = tf.reshape(hidden[0], (self.batch_size, self.dec_units))
        else:
            initial_state = self.hidFc(hidden[0])

        # add dropout
        prev_output = self.Dropout(prev_output)

        # reshape to pass through GRU
        prev_output = tf.expand_dims(prev_output, axis=1)

        # pass through GRU
        prev_output, state = self.GRU(prev_output, initial_state=initial_state)

        for i in range(self.num_layers - 1):

            # create initial state
            if hidden[i + 1].shape[1] == self.dec_units:
                initial_state = tf.reshape(hidden[i + 1], (self.batch_size, self.dec_units))
            else:
                initial_state = self.hidFc(hidden[i + 1])

            # add dropout and batchNorm
            prev_output = self.batchNorm(prev_output)
            prev_output = self.Dropout(prev_output)

            # pass through GRU
            prev_output, state = self.GRU(prev_output, initial_state=initial_state)

        # concatenate context vector with prev_output
        prev_output = tf.concat([tf.expand_dims(context_vector, 1), prev_output], axis=-1)

        # pass through final GRU layer
        output, state = self.GRU(prev_output)

        # reshape output
        output = tf.reshape(output, (-1, output.shape[2]))

        # add a dense layer for output
        output = self.Fc(output)

        return output, state, attention_weights
