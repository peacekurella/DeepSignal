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
import dataUtils.getTrajectory as traj
import numpy as np

# set up flags
FLAGS = flags.FLAGS

flags.DEFINE_string('input', '../train', 'Input Directory')
flags.DEFINE_string('ckpt', '../ckpt', 'Directory to store checkpoints')
flags.DEFINE_string('logs', '../logs', 'Directory to log metrics')
flags.DEFINE_boolean('load_ckpt', False, "Resume training")

flags.DEFINE_integer('keypoints', 57, 'Number of keypoints')
flags.DEFINE_integer('enc_size', 512, 'Hidden units in Encoder RNN')
flags.DEFINE_integer('dec_size', 512, 'Hidden units in Encoder RNN')
flags.DEFINE_integer('enc_layers', 1, 'Number of layers in encoder')
flags.DEFINE_integer('dec_layers', 1, 'Number of layers in decoder')
flags.DEFINE_float('enc_drop', 0.2, 'Encoder dropout probability')
flags.DEFINE_float('dec_drop', 0.2, 'Decoder dropout probability')
flags.DEFINE_integer('inp_length', 16, 'Input Sequence length')
flags.DEFINE_integer('pen_length', 0, 'penalty length')
flags.DEFINE_float('pen_smoothing', 5, 'Penalty smoothing coefficent')
flags.DEFINE_boolean('auto', False, 'Enable Auto Regression')

flags.DEFINE_integer('epochs', 60, 'Number of training epochs')
flags.DEFINE_integer('buffer_size', 5000, 'shuffle buffer size')
flags.DEFINE_integer('batch_size', 64, 'Mini batch size')
flags.DEFINE_integer('save', 10, 'Checkpoint save epochs')
flags.DEFINE_float('learning_rate', 0.0001, 'learning rate')


@tf.function
def train_step(encoder_input_seq, decoder_input_seq, target_seq, encoder, decoder, optimizer):
    """
    Defines a backward pass through the network
    :param encoder_input_seq: input sequence to the encoder of shape (batch_size, inp_length, 2*input_dim)
    :param encoder_input_seq: input sequence to the decoder of shape (batch_size, time_steps, 2*input_dim)
    :param target_seq: target sequence to the decoder of shape (batch_size, time_steps, input_dim)
    :param encoder: Encoder object
    :param decoder: Decoder object
    :param optimizer: optimizer object
    :return: batch loss for the given mini batch
    """

    # initialize loss
    loss = 0
    time_steps = target_seq.shape[1]
    predictions = []

    # initialize encoder hidden state
    enc_hidden = encoder.initialize_hidden_state(FLAGS.batch_size)

    with tf.GradientTape() as tape:

        # pass through encoder
        enc_output, enc_hidden = encoder(encoder_input_seq, enc_hidden, True)

        # first input to decoder
        if FLAGS.auto:
            # concatenate the last time step of encoder input
            ls = encoder_input_seq[:, -1, :FLAGS.keypoints]
            brs = decoder_input_seq[:, 0]
            dec_input = tf.concat([ls, brs], axis=1)
        else:
            dec_input = decoder_input_seq[:, 0]

        # input the hidden state
        dec_hidden = enc_hidden

        # start teacher forcing the network
        for t in range(1, time_steps):

            # pass dec_input and target sequence to decoder
            prediction, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output, True)
            predictions.append(prediction)

            # calculate the loss for every time step
            losses = tf.keras.losses.MSE(target_seq[:, t], prediction)
            loss += tf.reduce_mean(losses)

            # purge the tensors from memory
            del dec_input

            # set the next target value as input to decoder
            if FLAGS.auto:
                ls = target_seq[:, t-1]
                brs = decoder_input_seq[:, t]
                dec_input = tf.concat([ls, brs], axis=1)
            else:
                dec_input = decoder_input_seq[:, t]

    # calculate average batch loss
    batch_loss = (loss / time_steps)

    # get trainable variables
    variables = encoder.trainable_variables + decoder.trainable_variables

    # get the gradients
    gradients = tape.gradient(loss, variables)

    # purge tape from memory
    del tape

    # apply gradients to variables
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


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
    inp_length = FLAGS.inp_length
    learning_rate = FLAGS.learning_rate
    save = FLAGS.save
    logs = FLAGS.logs
    auto = FLAGS.auto

    # create encoder, decoder, and optimizer
    encoder = model.Encoder(enc_size, batch_size, enc_layers, enc_drop)
    decoder = model.Decoder(keypoints, dec_size, batch_size, dec_layers, dec_drop)
    optimizer = tf.keras.optimizers.Adam(
        lr=learning_rate
    )

    # create checkpoint saver
    checkpoint = tf.train.Checkpoint(
        optimizer=optimizer,
        encoder=encoder,
        decoder=decoder
    )

    # load checkpoint if needed
    if FLAGS.load_ckpt:
        checkpoint.restore(
            tf.train.latest_checkpoint(FLAGS.ckpt)
        )

    # set up summary writers
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(os.path.join(logs, current_time),  'train')
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    # start training
    for epoch in range(epochs):

        # measure start time
        start = time.time()
        epoch_loss = 0

        # do the actual training
        for (batch, (b, l, r)) in enumerate(dataset.shuffle(batch_size).take(steps_per_epoch)):

            # concatenate all three vectors
            input_tensor = tf.concat([l, b, r], axis=2)

            # split into input and target
            if auto:
                encoder_input_seq = input_tensor[:, :inp_length]
            else:
                encoder_input_seq = input_tensor[:, :inp_length, keypoints:]

            #  create decoder input and target sequence
            decoder_input_seq = input_tensor[:, inp_length:, keypoints:]
            target_seq = input_tensor[:, inp_length:, :keypoints]


            # actual train step
            batch_loss = train_step(encoder_input_seq, decoder_input_seq, target_seq, encoder, decoder, optimizer)
            epoch_loss += batch_loss

            # print progress periodically
            # save logs for tensorboard
            if batch % 100 == 0:
                progress = "Epoch {} Batch {} Loss {:.4f}".format(epoch + 1, batch, batch_loss.numpy())
                print(progress)

        # save checkpoint
        if (epoch + 1) % save == 0:
            ckpt_prefix = os.path.join(FLAGS.ckpt, "ckpt")
            checkpoint.save(ckpt_prefix)

        # log the loss
        with train_summary_writer.as_default():
            tf.summary.scalar('MSE', data=batch_loss.numpy(), step=epoch)

        # print progress
        progress = "\tAvg Epoch {} Loss {:.4f}".format(epoch + 1, (epoch_loss / steps_per_epoch))
        print(progress)
        print("\tEpoch train time {}".format(time.time() - start))


if __name__ == '__main__':
    app.run(main)
