import tensorflow as tf
from absl import app
from absl import flags
import os
import modelBuilder as model
import numpy as np
import sys

sys.path.append('..')
import dataUtils.dataVis as vis
import dataUtils.getData as db
import dataUtils.getTrajectory as traj
import csv

# set up flags
FLAGS = flags.FLAGS

flags.DEFINE_string('input', '../test', 'Input Directory')
flags.DEFINE_string('output', '../inference', 'Input Directory')
flags.DEFINE_string('ckpt', '../ckpt', 'Directory to store checkpoints')
flags.DEFINE_string('ckpt_ae', '../ckpt_ae', 'Directory to store checkpoints')
flags.DEFINE_boolean('load_ae', False, "Test auto encoder")

flags.DEFINE_integer('keypoints', 57, 'Number of keypoints')
flags.DEFINE_integer('enc_size', 512, 'Hidden units in Encoder RNN')
flags.DEFINE_integer('dec_size', 512, 'Hidden units in Encoder RNN')
flags.DEFINE_integer('enc_layers', 3, 'Number of layers in encoder')
flags.DEFINE_integer('dec_layers', 3, 'Number of layers in decoder')
flags.DEFINE_float('enc_drop', 0.2, 'Encoder dropout probability')
flags.DEFINE_float('dec_drop', 0.2, 'Decoder dropout probability')
flags.DEFINE_integer('inp_length', 17, 'Input Sequence length')

flags.DEFINE_integer('epochs', 60, 'Number of training epochs')
flags.DEFINE_integer('buffer_size', 5000, 'shuffle buffer size')
flags.DEFINE_integer('batch_size', 64, 'Mini batch size')


def run_inference(input_seq, target_seq, encoder, decoder):
    """
    Returns the predictions given input_seq
    :param input_seq: encoder input sequence
    :param target_seq: target sequence
    :param encoder: encoder object
    :param decoder: decoder object
    :return: predictions tensor of shape (batch_size, seqLength, input_size)
    """

    # number of time steps the
    time_steps = target_seq.shape[1]

    # initialize encoder hidden state
    enc_hidden = encoder.initialize_hidden_state(1)

    # encoder output
    enc_output, enc_hidden = encoder(input_seq, enc_hidden, False)

    # set the decoder hidden state and input
    dec_input = target_seq[:, 0]
    dec_hidden = enc_hidden

    # list of predictions
    predictions = []
    predictions.append(dec_input)

    for t in range(1, time_steps):
        # get the predictions
        prediction, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output, False)
        predictions.append(prediction)

        # update inputs to decoder
        del dec_input
        dec_input = prediction

    return tf.concat(predictions, axis=0)


def main(args):
    """
    Runs the inference example
    :param args:
    :return: None
    """

    # prep the input for inference
    # prepare the dataset
    input = FLAGS.input
    buffer_size = FLAGS.buffer_size
    batch_size = FLAGS.batch_size
    dataset = db.prepare_dataset(input, buffer_size, batch_size, False)
    if FLAGS.load_ae:
        checkpoint_dir = tf.train.latest_checkpoint(FLAGS.ckpt_ae)
    else:
        checkpoint_dir = tf.train.latest_checkpoint(FLAGS.ckpt)

    # set up experiment
    keypoints = FLAGS.keypoints
    enc_size = FLAGS.enc_size
    dec_size = FLAGS.dec_size
    enc_layers = FLAGS.enc_layers
    dec_layers = FLAGS.dec_layers
    enc_drop = FLAGS.enc_drop
    dec_drop = FLAGS.dec_drop
    batch_size = FLAGS.batch_size
    output = FLAGS.output

    # create encoder, decoder, and optimizer
    encoder = model.Encoder(enc_size, batch_size, enc_layers, enc_drop)
    decoder = model.Decoder(keypoints, dec_size, batch_size, dec_layers, dec_drop)

    # create checkpoint saver
    checkpoint = tf.train.Checkpoint(
        encoder=encoder,
        decoder=decoder
    )

    # restore checkpoints
    try:
        checkpoint.restore(checkpoint_dir).expect_partial()
    except:
        print("Failed to load checkpoint")
        sys.exit(0)

    # set the batch size to 1
    encoder.batch_size = 1
    decoder.batch_size = 1

    # metrics initialization
    total_error = []
    seqWise = {}
    count = 0

    # run a test prediction
    for (b, l, r) in dataset.take(5000):

        # unpack the sequences
        b_start, b = b
        l_start, l = l
        r_start, r = r

        for i in range(b.shape[0]):

            # split into input and target
            if not FLAGS.load_ae:
                input_seq = tf.expand_dims(tf.concat([b, l], axis=2)[i], axis=0)
            else:
                input_seq = tf.expand_dims(r[i], axis=0)
            target_seq = tf.expand_dims(r[i], axis=0)

            # run the inference op
            predictions = run_inference(input_seq, target_seq, encoder, decoder)

            # convert to absolute values
            buyer = traj.convert_to_absolute(b_start[i].numpy(), b[i].numpy())
            left = traj.convert_to_absolute(l_start[i].numpy(), l[i].numpy())
            right = traj.convert_to_absolute(r_start[i].numpy(), r[i].numpy())

            # convert trajectory to prediction
            pred_start = r_start[i]
            predictions = traj.convert_to_absolute(pred_start, predictions.numpy())

            # calculate the average joint error across all frames
            total_errors = tf.keras.losses.MSE(target_seq, predictions)
            total_errors = tf.reduce_mean(total_errors).numpy()

            # append for calculating metrics
            total_error.append(total_errors / (keypoints * l.shape[1]))
            seqWise.update({'epoch_' + str(count) + 'batch_' + str(i) + '_input': total_errors})

            # save the inference
            vis.create_animation(
                buyer.T,
                left.T,
                right.T,
                predictions.T,
                os.path.join(output, 'epoch_' + str(count) + 'batch_' + str(i) + '_output')
            )
            count += 1

    # print the average error, std over all batches
    print(np.mean(total_error))
    print(np.std(total_error))

    # write to csv file
    w = csv.writer(open("Errors.csv", "w"))
    for key, val in seqWise.items():
        w.writerow([key, val])


if __name__ == '__main__':
    app.run(main)
