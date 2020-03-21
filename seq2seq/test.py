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
import csv

# set up flags
FLAGS = flags.FLAGS

flags.DEFINE_string('input', '../test', 'Input Directory')
flags.DEFINE_string('output', '../inference', 'Input Directory')
flags.DEFINE_string('ckpt', '../ckpt', 'Directory to store checkpoints')

flags.DEFINE_integer('keypoints', 57, 'Number of keypoints')
flags.DEFINE_integer('enc_size', 512, 'Hidden units in Encoder RNN')
flags.DEFINE_integer('dec_size', 512, 'Hidden units in Encoder RNN')
flags.DEFINE_integer('enc_layers', 1, 'Number of layers in encoder')
flags.DEFINE_integer('dec_layers', 1, 'Number of layers in decoder')
flags.DEFINE_float('enc_drop', 0.2, 'Encoder dropout probability')
flags.DEFINE_float('dec_drop', 0.2, 'Decoder dropout probability')
flags.DEFINE_integer('inp_length', 17, 'Input Sequence length')
flags.DEFINE_boolean('auto', False, 'Enable Auto Regression')

flags.DEFINE_integer('epochs', 60, 'Number of training epochs')
flags.DEFINE_integer('buffer_size', 5000, 'shuffle buffer size')
flags.DEFINE_integer('batch_size', 64, 'Mini batch size')

def run_inference(input_seq, target_seq, encoder, decoder):
    """
    Returns the predictions given input_seq
    :param input_seq: input sequence
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

    # first input to decoder
    if FLAGS.auto:
        buyer = input_seq[:, -1, :decoder.output_size]
        sellers = target_seq[:, 0, decoder.output_size:]
        dec_input = tf.concat([buyer, sellers], axis=1)
    else:
        dec_input = target_seq[:, 0, decoder.output_size:]
    dec_hidden = enc_hidden

    # list of predictions
    predictions = []

    for t in range(0, time_steps):
        # get the predictions
        prediction, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output, False)
        predictions.append(tf.expand_dims(prediction, axis=1))

        # update inputs to decoder
        del dec_input
        if FLAGS.auto:
            buyer = prediction
            sellers = target_seq[:, t, decoder.output_size:]
            dec_input = tf.concat([buyer, sellers], axis=1)
        else:
            dec_input = target_seq[:, t, decoder.output_size:]

    return tf.squeeze(tf.concat(predictions, axis=1), axis=0)


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
    inp_length = FLAGS.inp_length
    output = FLAGS.output
    auto = FLAGS.auto

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

    # metrics initialization
    total_error = []
    seqWise = {}
    count = 0

    # run a test prediction
    for (b, l, r) in dataset.take(5000):

        for i in range(b.shape[0]):
            # create a target animation
            vis.create_animation(
                tf.transpose(b[i, inp_length:]).numpy(),
                tf.transpose(l[i, inp_length:]).numpy(),
                tf.transpose(r[i, inp_length:]).numpy(),
                os.path.join(output, 'epoch_'+str(count)+'batch_'+str(i)+'_input')
            )

            # concatenate all three vectors
            input_tensor = tf.expand_dims(tf.concat([b, l, r], axis=2)[i], axis=0)

            # split into input and target
            if auto:
                input_seq = input_tensor[:, :inp_length]
            else:
                input_seq = input_tensor[:, :inp_length, keypoints:]
            target_seq = input_tensor[:, inp_length:]

            # run the inference op
            buyers = run_inference(input_seq, target_seq, encoder, decoder)

            # calculate the average joint error across all frames
            total_errors = tf.keras.losses.MSE(b[:, inp_length:], buyers)
            total_errors = tf.reduce_mean(total_errors).numpy()

            # append for calculating metrics
            total_error.append( total_errors / (keypoints * buyers.shape[0]))
            seqWise.update({'epoch_'+str(count)+'batch_'+str(i)+'_input': total_errors})

            # save the inference
            vis.create_animation(
                tf.transpose(buyers).numpy(),
                tf.transpose(l[i, inp_length:]).numpy(),
                tf.transpose(r[i, inp_length:]).numpy(),
                os.path.join(output, 'epoch_'+str(count)+'batch_'+str(i)+'_output')
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
