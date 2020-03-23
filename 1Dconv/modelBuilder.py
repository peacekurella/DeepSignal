import tensorflow as tf

# configuration changes for RTX enabled devices
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

class motionAutoEncoder(tf.keras.Model):

    def __init__(self, dropout_rate, keypoints):
        """
        initializes 1D conv-deconv autoencoder
        :param dropout_rate: dropout rate
        :param keypoints: output keypoints
        """

        # call the super function
        super(motionAutoEncoder, self).__init__()

        # define the encoder
        self.Encoder = tf.keras.Sequential([

            tf.keras.layers.BatchNormalization(
                center=False,
                scale=False
            ),
            tf.keras.layers.Dropout(
                dropout_rate
            ),
            tf.keras.layers.Conv1D(
                256,
                kernel_size=3,
                activation='relu',
                padding="causal"
            ),
            tf.keras.layers.Dropout(
                dropout_rate
            ),
            tf.keras.layers.Conv1D(
                256,
                kernel_size=3,
                activation='relu',
                padding="causal"
            ),
            tf.keras.layers.Dropout(
                dropout_rate
            ),
            tf.keras.layers.Conv1D(
                256,
                kernel_size=3,
                activation='relu',
                padding="causal"
            ),
            tf.keras.layers.Dropout(
                dropout_rate
            ),
            tf.keras.layers.MaxPool1D(
                 pool_size=2,
                 strides=2
            ),
            tf.keras.layers.Dropout(
                dropout_rate
            )
        ])

        # define the decoder
        self.Decoder = tf.keras.Sequential([

            tf.keras.layers.Conv2DTranspose(
                keypoints,
                kernel_size=(1,25),
                strides=2,
                data_format='channels_last',
                activation='relu'
            )

        ])

    def call(self, inputs, training=None, mask=None):

        encoder_output = self.Encoder(inputs, training=training)
        encoder_output = tf.expand_dims(encoder_output, axis=-1)
        decoder_output = self.Decoder(encoder_output, training=training)
        return decoder_output[:, :, 0, :]



