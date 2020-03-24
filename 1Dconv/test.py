import sys
sys.path.append("..")
import modelBuilder as modelBuilder
import tensorflow as tf
from absl import flags
from absl import app
from dataUtils import getData as db
import dataUtils.dataVis as vis
import os
import numpy as np
import csv

# set up flags
FLAGS = flags.FLAGS

flags.DEFINE_string('input', '../train', 'Input Directory')
flags.DEFINE_string('output', '../convOut', 'output Directory')
flags.DEFINE_string('ckpt', '../ckpt', 'Directory to store checkpoints')
flags.DEFINE_string('logs', '../logs', 'Directory to log metrics')

flags.DEFINE_integer('epochs', 60, 'Number of training epochs')
flags.DEFINE_float('dropout', 0.25, 'dropout probability')
flags.DEFINE_integer('keypoints', 57, 'Number of keypoints')
flags.DEFINE_integer('buffer_size', 5000, 'shuffle buffer size')
flags.DEFINE_integer('batch_size', 64, 'Mini batch size')
flags.DEFINE_integer('save', 10, 'Checkpoint save epochs')
flags.DEFINE_integer('seqLength', 56, 'sequence to be passed for predictions')



def run_inference(input, motionEncoder, motionDecoder):
    """
    Returns the predictions
    :param input: input frames to the model
    :param motionEncoder: 1D conv motion Encoder
    :param motionDecoder: 1D conv motion Decoder
    :return: output frames from the model
    """
    encoder_out = motionEncoder(input, training=False)
    predictions = motionDecoder(encoder_out, training=False)
    return predictions

def main(args):
    """
    Runs the inference examples
    :param args:
    :return: None
    """

    # prepare the dataset
    input = FLAGS.input
    buffer_size = FLAGS.buffer_size
    batch_size = FLAGS.batch_size
    dataset = db.prepare_dataset(input, buffer_size, batch_size, True)

    # set up experiment
    steps_per_epoch = dataset.reduce(0, lambda x, _: x + 1).numpy()
    dropout_rate = FLAGS.dropout
    keypoints = FLAGS.keypoints
    checkpoint_dir = tf.train.latest_checkpoint(FLAGS.ckpt)
    seqLength = FLAGS.seqLength
    output = FLAGS.output

    # set up the model
    motionEncoder = modelBuilder.motionEncoder(dropout_rate)
    motionDecoder = modelBuilder.motionDecoder(keypoints)

    # create checkpoint saver
    checkpoint = tf.train.Checkpoint(
        motionDecoder = motionDecoder,
        motionEncoder = motionEncoder
    )

    # restore checkpoints
    try:
        checkpoint.restore(checkpoint_dir).expect_partial()
    except:
        print("Failed to load checkpoint")
        sys.exit(0)

    # metrics initialization
    total_error = []

    # run a test prediction
    for (b, l, r) in dataset.take(5000):

        # input and target sequences
        input = tf.concat([b[:, :seqLength, :], r[:, :seqLength, :]], axis=2)
        target = l[:, :seqLength, :]

        # run the inference op
        ls = run_inference(input, motionEncoder, motionDecoder)
        print(ls.shape)

        # calculate the average joint error across all frames
        total_errors = tf.keras.losses.MSE(target, ls)
        total_errors = tf.reduce_mean(total_errors).numpy()

        # append for calculating metrics
        total_error.append( total_errors / (keypoints * l.shape[0]))

        count = 0
        for i in range(b.shape[0]):

            # save the inference
            vis.create_animation(
                tf.transpose(b[i, :seqLength, :]).numpy(),
                tf.transpose(l[i, :seqLength, :]).numpy(),
                tf.transpose(r[i, :seqLength, :]).numpy(),
                tf.transpose(ls[i]).numpy(),
                os.path.join(output, 'epoch_'+str(count)+'batch_'+str(i)+'_output')
            )
            count += 1

    # print the average error, std over all batches
    print(np.mean(total_error))
    print(np.std(total_error))


if __name__ == '__main__':
    app.run(main)