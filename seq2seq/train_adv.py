import tensorflow as tf
import sys
import os
import modelBuilder as model
from absl import app
from absl import flags
import time
import datetime

sys.path.append('..')
import dataUtils.getData as db


# set up flags
FLAGS = flags.FLAGS

flags.DEFINE_string('input', '../train', 'Input Directory')
flags.DEFINE_string('ckpt', '../ckpt_adv', 'Directory to store checkpoints')
flags.DEFINE_string('ckpt_ae', '../ckpt_ae', 'Directory to retrieve AE checkpoints')
flags.DEFINE_string('logs', '../logs', 'Directory to log metrics')
flags.DEFINE_boolean('load_ae', False, 'Load the decoder from auto encoder')
flags.DEFINE_boolean('load_ckpt', False, 'Resume training')
flags.DEFINE_boolean('train_dec', True, 'Train the decoder also')

flags.DEFINE_integer('keypoints', 57, 'Number of keypoints')
flags.DEFINE_integer('enc_size', 512, 'Hidden units in Encoder RNN')
flags.DEFINE_integer('dec_size', 512, 'Hidden units in Encoder RNN')
flags.DEFINE_integer('disc_size', 512, 'Hidden units in Discriminator RNN')
flags.DEFINE_integer('enc_layers', 3, 'Number of layers in encoder')
flags.DEFINE_integer('dec_layers', 3, 'Number of layers in decoder')
flags.DEFINE_float('enc_drop', 0.2, 'Encoder dropout probability')
flags.DEFINE_float('dec_drop', 0.2, 'Decoder dropout probability')
flags.DEFINE_float('disc_drop', 0.2, 'Discriminator dropout probability')
flags.DEFINE_float('gen_smoothing', 100, 'Smoothing value to use for generator loss')

flags.DEFINE_integer('epochs', 140, 'Number of training epochs')
flags.DEFINE_integer('buffer_size', 5000, 'shuffle buffer size')
flags.DEFINE_integer('batch_size', 64, 'Mini batch size')
flags.DEFINE_integer('save', 10, 'Checkpoint save epochs')
flags.DEFINE_float('learning_rate', 0.0001, 'learning rate')


def discriminator_loss(y_hat_real, y_hat_gen):
    """
    Calculates the Discriminator loss
    :param y_hat_gen: predicted output from from discriminator for generated sequence
    :param y_hat_real: predicted output from discriminator for real sequence
    :return: total discriminator loss
    """

    # initialize the loss object
    binary_cross = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # calculate real and generated loss
    real_loss = binary_cross(tf.ones_like(y_hat_real), y_hat_real)
    generated_loss = binary_cross(tf.zeros_like(y_hat_gen), y_hat_gen)

    return real_loss + generated_loss


def generator_loss(disc_out, gen_out, target_seq):
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
    total_loss = adv_loss + (FLAGS.gen_smoothing * mse_loss)

    return adv_loss, mse_loss, total_loss


@tf.function
def train_step(input_seq, target_seq, encoder, decoder, discriminator, gen_optimizer, disc_optimizer):
    """
    Defines a backward pass through the network
    :param input_seq: input sequence to the encoder of shape (batch_size, inp_length, 2*keypoints)
    :param target_seq: target sequence to the decoder of shape (batch_size, time_steps, keypoints)
    :param encoder: Encoder object
    :param decoder: Decoder object
    :param discriminator: Discriminator object
    :param gen_optimizer: generator optimizer object
    :param disc_optimizer: discriminator optimizer object
    :return: disc_loss, adv_loss, mse_loss, gen_loss
    """

    # initialize loss
    time_steps = target_seq.shape[1]

    # initialize encoder hidden state
    enc_hidden = encoder.initialize_hidden_state(FLAGS.batch_size)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # pass through encoder
        print(input_seq.shape)
        enc_output, enc_hidden = encoder(input_seq, enc_hidden, True)

        # input the hidden state
        dec_hidden = enc_hidden
        dec_input = tf.zeros(target_seq[:, 0].shape)

        # gather the predictions
        predictions = []
        for t in range(time_steps):
            # pass dec_input and target sequence to decoder
            prediction, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output, True)
            predictions.append(tf.expand_dims(prediction, axis=1))

            # set the next target value as input to decoder
            dec_input = target_seq[:, t]

        # concatenate predictions time axis
        gen_seq = tf.concat(predictions, axis=1)

        # pass through discriminator
        disc_real = discriminator(target_seq)
        disc_gen = discriminator(gen_seq)

        # calculate discriminator loss
        disc_loss = discriminator_loss(disc_real, disc_gen)

        # calculate generator loss
        adv_loss, mse_loss, gen_loss = generator_loss(disc_gen, gen_seq, target_seq)

    # train generator
    if not FLAGS.train_dec:
        variables = encoder.trainable_variables + decoder.trainable_variables
    else:
        variables = encoder.trainable_variables
    gradients = gen_tape.gradient(gen_loss, variables)
    gen_optimizer.apply_gradients(zip(gradients, variables))

    # train discriminator
    variables = discriminator.trainable_variables
    gradients = disc_tape.gradient(disc_loss, variables)
    disc_optimizer.apply_gradients(zip(gradients, variables))

    # calculate average batch losses
    disc_loss = (disc_loss / time_steps)
    adv_loss = (adv_loss / time_steps)
    mse_loss = (mse_loss / time_steps)
    gen_loss = (gen_loss / time_steps)

    return disc_loss, adv_loss, mse_loss, gen_loss


def main(args):
    """
    Trains the model and save the checkpoints
    :param args: arguments to main
    :return: None
    """

    # prepare the dataset
    input = FLAGS.input
    buffer_size = FLAGS.buffer_size
    batch_size = FLAGS.batch_size
    dataset = db.prepare_dataset(input, buffer_size, batch_size, True)

    # set up experiment
    keypoints = FLAGS.keypoints
    enc_size = FLAGS.enc_size
    dec_size = FLAGS.dec_size
    enc_layers = FLAGS.enc_layers
    dec_layers = FLAGS.dec_layers
    enc_drop = FLAGS.enc_drop
    dec_drop = FLAGS.dec_drop
    batch_size = FLAGS.batch_size
    epochs = FLAGS.epochs
    steps_per_epoch = dataset.reduce(0, lambda x, _: x + 1).numpy()
    learning_rate = FLAGS.learning_rate
    save = FLAGS.save
    logs = FLAGS.logs
    disc_size = FLAGS.disc_size
    disc_drop = FLAGS.disc_drop

    # create encoder, decoder, discriminator, and optimizer
    encoder = model.Encoder(enc_size, batch_size, enc_layers, enc_drop)
    decoder = model.Decoder(keypoints, dec_size, batch_size, dec_layers, dec_drop)
    gen_optimizer = tf.keras.optimizers.Adam(
        lr=learning_rate
    )
    disc_optimizer = tf.keras.optimizers.Adam(
        lr=learning_rate
    )
    discriminator = model.Discriminator(disc_size, batch_size, disc_drop)

    # create checkpoint saver
    checkpoint = tf.train.Checkpoint(
        gen_optimizer=gen_optimizer,
        disc_optimizer=disc_optimizer,
        encoder=encoder,
        decoder=decoder,
        discriminator=discriminator
    )

    # load checkpoint if needed
    if FLAGS.load_ckpt:
        checkpoint.restore(
            tf.train.latest_checkpoint(FLAGS.ckpt)
        )

    # load the auto encoder's decoder if needed
    if FLAGS.load_ae:
        ae_checkpoint = tf.train.Checkpoint(
            decoder=decoder
        )
        ae_checkpoint.restore(
            tf.train.latest_checkpoint(FLAGS.ckpt_ae)
        )

    # set up summary writers
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(os.path.join(logs, current_time), 'train')
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    # start training
    for epoch in range(epochs):

        # measure start time
        start = time.time()
        epoch_disc_loss = 0
        epoch_adv_loss = 0
        epoch_mse_loss = 0
        epoch_gen_loss = 0

        # do the actual training
        batch = 0
        for (b, l, r) in dataset.shuffle(batch_size).take(steps_per_epoch):

            # discard start states
            _, b = b
            _, l = l
            _, r = r

            # split into input and target
            input_seq = tf.concat([b, r], axis=2)
            target_seq = l

            # actual train step
            disc_loss, adv_loss, mse_loss, gen_loss = train_step(
                input_seq,
                target_seq,
                encoder,
                decoder,
                discriminator,
                gen_optimizer,
                disc_optimizer
            )
            epoch_disc_loss += disc_loss
            epoch_adv_loss += adv_loss
            epoch_gen_loss += gen_loss
            epoch_mse_loss += mse_loss

            # print progress periodically
            # save logs for tensorboard
            if batch % 10 == 0:
                progress = "Epoch {} Batch {} Loss {:.4f}".format(epoch + 1, batch, gen_loss.numpy())
                print(progress)

            # increase batch size
            batch += 1

            # split into input and target
            input_seq = tf.concat([b, l], axis=2)
            target_seq = r

            # actual train step
            disc_loss, adv_loss, mse_loss, gen_loss = train_step(
                input_seq,
                target_seq,
                encoder,
                decoder,
                discriminator,
                gen_optimizer,
                disc_optimizer
            )
            epoch_disc_loss += disc_loss
            epoch_adv_loss += adv_loss
            epoch_gen_loss += gen_loss
            epoch_mse_loss += mse_loss

            # print progress periodically
            # save logs for tensorboard
            if batch % 10 == 0:
                progress = "Epoch {} Batch {} Loss {:.4f}".format(epoch + 1, batch, gen_loss.numpy())
                print(progress)

            # increase batch size
            batch += 1

        # save checkpoint
        if (epoch + 1) % save == 0:
            ckpt_prefix = os.path.join(FLAGS.ckpt, "ckpt")
            checkpoint.save(ckpt_prefix)

        # log the loss
        with train_summary_writer.as_default():
            tf.summary.scalar('MSE LOSS', data=(epoch_mse_loss / 2), step=epoch)
            tf.summary.scalar('ADV LOSS', data=(epoch_adv_loss / 2), step=epoch)
            tf.summary.scalar('GAN LOSS', data=(epoch_gen_loss / 2), step=epoch)
            tf.summary.scalar('DISC LOSS', data=(epoch_disc_loss / 2), step=epoch)

        # print progress
        progress = "\tAvg Epoch {} Loss {:.4f}".format(epoch + 1, (epoch_gen_loss / 2))
        print(progress)
        print("\tEpoch train time {}".format(time.time() - start))


if __name__ == '__main__':
    app.run(main)
