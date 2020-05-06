import tensorflow as tf
import numpy as np
import sys
import os
from absl import app
from absl import flags
from Net.ModelZoo.AutoEncoderMSE import AutoEncoderMSE
from Net.ModelZoo.ConcatenationEncoderDecoderMSE import ConcatenationEncoderDecoderMSE
from Net.ModelZoo.TransformationEncoderDecoderMSE import TransformationEncoderDecoderMSE
from Net.ModelZoo.TransformationEncoderMSEAE import TransformationEncoderMSEAE
from DataUtils.DataGenerator import DataGenerator
from DataUtils.TrajectoryHandler import convert_to_absolute
from DataUtils.InputStandardizer import InputStandardizer
from DataUtils.DataVisualizer import DataVisualizer

# set up flags
FLAGS = flags.FLAGS

flags.DEFINE_string('input', 'Data/test', 'Input Directory')
flags.DEFINE_string('output', 'Data/inference', 'output directory')
flags.DEFINE_string('ckpt', 'Data/Checkpoints', 'Directory to store checkpoints')
flags.DEFINE_string('stats', 'Data/stats', 'Directory to store stats')
flags.DEFINE_boolean('standardize_data', True, 'standardize data before training')

flags.DEFINE_string('model_code', 'TEDM', 'Defines the model to load')
flags.DEFINE_integer('output_dims', 57, 'Number of key points in output')
flags.DEFINE_integer('enc_size', 512, 'Hidden units in Encoder RNN')
flags.DEFINE_integer('dec_size', 512, 'Hidden units in Encoder RNN')
flags.DEFINE_integer('disc_size', 512, 'Hidden units in Discriminator RNN')
flags.DEFINE_integer('enc_layers', 3, 'Number of layers in encoder')
flags.DEFINE_integer('dec_layers', 3, 'Number of layers in decoder')

flags.DEFINE_integer('batch_size', 64, 'Mini batch size')


def get_input_target(b, l, r):
    """
    generates input and target sequences for the models
    :param b: buyer trajectory
    :param l: left seller trajectory
    :param r: right seller trajectory
    :return: input_seq1, target_seq1, input_seq2, target_seq2
    """

    # initialize inputs and targets
    input_seq1 = None
    input_seq2 = None
    target_seq1 = None
    target_seq2 = None

    if FLAGS.model_code == "AE":
        input_seq1 = l
        target_seq1 = l
        input_seq2 = r
        target_seq2 = r

    if FLAGS.model_code in ["CEDM"]:
        input_seq1 = tf.concat([b, r], axis=2)
        target_seq1 = l
        input_seq2 = tf.concat([b, l], axis=2)
        target_seq2 = r

    elif FLAGS.model_code in["TEDM", "TEMA"]:
        input_seq1 = (b, r)
        target_seq1 = l
        input_seq2 = (b, l)
        target_seq2 = r

    return input_seq1, target_seq1, input_seq2, target_seq2


def get_model():
    """
    returns the appropriate model
    :return:
    """

    if FLAGS.model_code == "AE":
        return AutoEncoderMSE(
            FLAGS.enc_size,
            FLAGS.batch_size,
            FLAGS.enc_layers,
            0,
            FLAGS.dec_size,
            FLAGS.output_dims,
            FLAGS.dec_layers,
            0,
            1
        )

    elif FLAGS.model_code == "CEDM":
        return ConcatenationEncoderDecoderMSE(
            FLAGS.enc_size,
            FLAGS.batch_size,
            FLAGS.enc_layers,
            0,
            FLAGS.dec_size,
            FLAGS.output_dims,
            FLAGS.dec_layers,
            0,
            1,
            os.path.join(FLAGS.ckpt, 'AE')
        )

    elif FLAGS.model_code == "TEDM":
        return TransformationEncoderDecoderMSE(
            FLAGS.enc_size,
            FLAGS.batch_size,
            FLAGS.enc_layers,
            0,
            FLAGS.dec_size,
            FLAGS.output_dims,
            FLAGS.dec_layers,
            0,
            1,
            os.path.join(FLAGS.ckpt, 'AE')
        )

    elif FLAGS.model_code == "TEMA":
        return TransformationEncoderMSEAE(
            FLAGS.enc_size,
            FLAGS.batch_size,
            FLAGS.enc_layers,
            0,
            FLAGS.dec_size,
            FLAGS.output_dims,
            FLAGS.dec_layers,
            0,
            1,
            os.path.join(FLAGS.ckpt, 'AE')
        )


def main(args):
    """
    Runs the inference example
    :param args:
    :return: None
    """

    # prep the input for inference
    model = get_model()
    model.load_model(
        os.path.join(FLAGS.ckpt, FLAGS.model_code)
    )

    # get the data generator object
    data_generator = DataGenerator(
        FLAGS.standardize_data,
        FLAGS.stats
    )

    # prepare the dataset
    dataset = data_generator.prepare_dataset(
        FLAGS.input,
        5000,
        FLAGS.batch_size,
        False
    )

    # calculate the number of steps per each epoch
    steps_per_epoch = dataset.reduce(0, lambda x, _: x + 1).numpy()

    # initialize standardizer
    standardizer = InputStandardizer(
        FLAGS.stats
    )

    # metrics initialization
    total_error = []
    count = 0

    # start the inference
    for b, l, r in dataset.take(steps_per_epoch):

        # unpack the sequences
        b_start, b = b
        l_start, l = l
        r_start, r = r

        # create input and target sequences
        input_seq1, target_seq1, input_seq2, target_seq2 = get_input_target(b, l, r)

        # run the inference op
        predictions1 = model.run_inference(input_seq1, target_seq1.shape[1], target_seq1[:, 0].shape)
        predictions2 = model.run_inference(input_seq2, target_seq2.shape[1], target_seq2[:, 0].shape)

        for i in range(b.shape[0]):

            # de standardize
            if FLAGS.standardize_data:
                b_std = standardizer.destandardize(b[i].numpy())
                pred1 = standardizer.destandardize(predictions1[i].numpy())
                pred2 = standardizer.destandardize(predictions2[i].numpy())
                target1 = standardizer.destandardize(target_seq1[i].numpy())
                target2 = standardizer.destandardize(target_seq2[i].numpy())
            else:
                b_std = b[i].numpy()
                pred1 = predictions1[i].numpy()
                pred2 = predictions2[i].numpy()
                target1 = target_seq1[i].numpy()
                target2 = target_seq2[i].numpy()

            # convert to absolute values
            b_std = convert_to_absolute(b_start[i], b_std)
            pred1 = convert_to_absolute(l_start[i], pred1)
            pred2 = convert_to_absolute(r_start[i], pred2)
            target1 = convert_to_absolute(l_start[i], target1)
            target2 = convert_to_absolute(r_start[i], target2)

            joints = [b_std.T, target1.T, target2.T, pred1.T, pred2.T]

            # save the animation
            filename = os.path.join(os.path.join(FLAGS.output, str(count)), 'prediction')

            # create output directory if doesnt exist
            if not os.path.isdir(os.path.join(FLAGS.output, str(count))):
                try:
                    os.mkdir(os.path.join(FLAGS.output, str(count)))
                except:
                    print("Error creating output directory")
                    sys.exit()

            # initialize the visualizer
            data_visualizer = DataVisualizer()
            data_visualizer.create_animation(joints, filename)
            count += 1

            # calculate the average joint error across all frames
            total_errors1 = tf.keras.losses.MSE(target1, pred1)
            total_errors1 = tf.reduce_mean(total_errors1).numpy()
            total_errors2 = tf.keras.losses.MSE(target2, pred2)
            total_errors2 = tf.reduce_mean(total_errors2).numpy()

            # append for calculating metrics
            total_error.append(total_errors1 + total_errors2 / (b_std.shape[0] * b_std.shape[1] * 2))

    # save the stats
    np.savetxt(
        os.path.join(FLAGS.stats, 'mean_error.txt'),
        np.mean(total_error)
    )

    np.savetxt(
        os.path.join(FLAGS.stats, 'mean_std.txt'),
        np.std(total_error)
    )


if __name__ == '__main__':
    app.run(main)
