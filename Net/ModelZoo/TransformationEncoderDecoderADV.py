from Net.BaseModels.TransformationEncoder import TransformationEncoder
from Net.BaseModels.SequenceDecoder import SequenceDecoder
from Net.BaseModels.SequenceDiscriminator import SequenceDiscriminator
import tensorflow as tf
import os
import sys

# configuration changes for RTX enabled devices
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


class TransformationEncoderDecoderADV:
    """
    Concatenated state input, sequence reconstruction using ADV loss
    """

    def __init__(self, enc_units, batch_size, enc_layers, enc_dropout_rate,
                 dec_units, output_size, dec_layers, dec_dropout_rate,
                 learning_rate, auto_encoder_dir, disc_units, disc_dropout_rate, gen_smoothing):
        """
        Initializes the ConcatenationEncoderDecoderADV class with MSE loss
        :param enc_units: Number of hidden units in encoder
        :param batch_size: Batch size for training
        :param enc_layers: Number of layers in the Encoder
        :param enc_dropout_rate: Encoder Dropout rate
        :param dec_units: Number of hidden units in Decoder
        :param output_size: Number of output key points
        :param dec_layers: Number of layers in the Decoder
        :param dec_dropout_rate: Decoder Dropout rate
        :param auto_encoder_dir: checkpoint_dir for auto_encoder
        :param learning_rate: learning rate
        :param disc_dropout_rate: Discriminator dropout rate
        :param gen_smoothing: MSE Smoothing value
        :param disc_units: discriminator units
        """

        # create encoder
        self.encoder = TransformationEncoder(
            enc_units,
            batch_size,
            enc_layers,
            enc_dropout_rate
        )

        # create decoder
        self.decoder = SequenceDecoder(
            output_size,
            dec_units,
            batch_size,
            dec_layers,
            dec_dropout_rate
        )

        # create discriminator
        self.discriminator = SequenceDiscriminator(
            disc_units,
            batch_size,
            disc_dropout_rate
        )

        # create optimizers
        self.gen_optimizer = tf.keras.optimizers.Adam(
            lr=learning_rate
        )
        self.disc_optimizer = tf.keras.optimizers.Adam(
            lr=learning_rate/10
        )

        # create checkpoint saver
        self.checkpoint = tf.train.Checkpoint(
            gen_optimizer=self.gen_optimizer,
            disc_optimizer=self.disc_optimizer,
            encoder=self.encoder,
            decoder=self.decoder,
            discriminator=self.discriminator
        )

        # load the decoder from auto encoder
        auto_encoder = tf.train.Checkpoint(
            decoder=self.decoder
        )

        try:
            auto_encoder.restore(
                tf.train.latest_checkpoint(auto_encoder_dir)
            ).expect_partial()
        except:
            print("Loading decoder failed")
            sys.exit(0)

        # load the smoothing parameter
        self.gen_smoothing = gen_smoothing

    def save_model(self, checkpoint_dir):
        """
        Saves the model checkpoint
        :param checkpoint_dir: Directory to save the checkpoint in
        :return:
        """
        ckpt_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.checkpoint.save(ckpt_prefix)

    def load_model(self, checkpoint_dir):
        """
        Loads the model from the checkpoint
        :param checkpoint_dir: Directory in which the checkpoint is stored
        :return:
        """
        self.checkpoint.restore(
            tf.train.latest_checkpoint(checkpoint_dir)
        )

    def discriminator_loss(self, y_hat_real, y_hat_gen):
        """
        Calculates the SequenceDiscriminator loss
        :param y_hat_gen: predicted output from from discriminator for generated sequence
        :param y_hat_real: predicted output from discriminator for real sequence
        :return: real_loss, generator_loss, total discriminator loss
        """

        # initialize the loss object
        binary_cross = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        # calculate real and generated loss
        real_loss = binary_cross(tf.ones_like(y_hat_real), y_hat_real)
        generated_loss = binary_cross(tf.zeros_like(y_hat_gen), y_hat_gen)

        return real_loss, generated_loss, real_loss + generated_loss

    def generator_loss(self, disc_out, gen_out, target_seq):
        """
        Calculates the Generator loss
        :param disc_out: Predicted output from discriminator for the generator's output
        :param gen_out: Generator output
        :param target_seq: target sequence for the generator output
        :return: adverserial loss, MSE loss, total loss
        """

        # initialize the loss objects
        binary_cross = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        # calculate adverserial loss
        adv_loss = binary_cross(tf.ones_like(disc_out), disc_out)

        # calculate MSE loss
        mse_loss = tf.reduce_mean(tf.pow(target_seq - gen_out, 2))

        # calculate total loss
        total_loss = adv_loss + (self.gen_smoothing * mse_loss)

        return adv_loss, mse_loss, total_loss

    @tf.function
    def train_step(self, input_seq, target_seq):
        """
        Defines a backward pass through the network
        :param input_seq:
        :param target_seq:
        :return:
        """

        # initialize loss
        time_steps = target_seq.shape[1]

        # initialize encoder hidden state
        enc_hidden = self.encoder.encoderA.initialize_hidden_state(self.encoder.encoderA.batch_size)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # pass through encoder
            enc_output, enc_hidden = self.encoder(input_seq, enc_hidden, True)

            # input the hidden state
            dec_hidden = enc_hidden
            dec_input = tf.zeros(target_seq[:, 0].shape)

            # gather the predictions
            predictions = []
            for t in range(time_steps):
                # pass dec_input and target sequence to decoder
                prediction, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output, True)
                predictions.append(tf.expand_dims(prediction, axis=1))

                # set the next target value as input to decoder
                dec_input = target_seq[:, t]

            # concatenate predictions time axis
            gen_seq = tf.concat(predictions, axis=1)

            # pass through discriminator
            disc_real = self.discriminator(target_seq)
            disc_gen = self.discriminator(gen_seq)

            # calculate discriminator loss
            real_loss, gen_disc_loss, disc_loss = self.discriminator_loss(disc_real, disc_gen)

            # calculate generator loss
            adv_loss, mse_loss, gen_loss = self.generator_loss(disc_gen, gen_seq, target_seq)

        variables = self.encoder.trainable_variables
        gradients = gen_tape.gradient(gen_loss, variables)
        self.gen_optimizer.apply_gradients(zip(gradients, variables))

        # train discriminator
        variables = self.discriminator.trainable_variables
        gradients = disc_tape.gradient(disc_loss, variables)
        self.disc_optimizer.apply_gradients(zip(gradients, variables))

        # calculate average batch losses
        real_loss = (real_loss / time_steps)
        gen_disc_loss = (gen_disc_loss / time_steps)
        disc_loss = (disc_loss / time_steps)
        adv_loss = (adv_loss / time_steps)
        mse_loss = (mse_loss / time_steps)
        gen_loss = (gen_loss / time_steps)

        loss_dict = {
            'TEDA_GEN_MSE':
                {
                    'Reconstruction Loss': mse_loss
                },
            'TEDA_GEN_ADV':
                {
                    'Generator Loss': adv_loss
                },
            'TEDA_GEN_TOTAL':
                {
                    'Generator Loss': gen_loss
                },
            'TEDA_DISC_REAL':
                {
                    'Discriminator Loss': real_loss
                },
            'TEDA_DISC_GEN':
                {
                    'Discriminator Loss': gen_disc_loss
                },
            'TEDA_DISC_TOTAL':
                {
                    'Discriminator Loss': disc_loss
                },
        }

        return loss_dict

    def run_inference(self, input_seq, time_steps, output_shape):
        """
        Returns the predictions given input_seq
        :param output_shape: shape of the decoder output
        :param time_steps: Number of time steps to run the inference operation for
        :param input_seq: encoder input sequence
        :return: predictions tensor of shape (batch_size, seqLength, input_size)
        """

        # initialize encoder hidden state
        enc_hidden = self.encoder.encoderA.initialize_hidden_state(self.encoder.encoderA.batch_size)

        # encoder output
        enc_output, enc_hidden = self.encoder(input_seq, enc_hidden, False)

        # set the decoder hidden state and input
        dec_input = tf.zeros(output_shape)
        dec_hidden = enc_hidden

        # list of predictions
        predictions = []

        for t in range(time_steps):
            # get the predictions
            prediction, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output, False)
            predictions.append(tf.expand_dims(prediction, axis=1))

            # update inputs to decoder
            del dec_input
            dec_input = prediction

        return tf.concat(predictions, axis=1)