import tensorflow as tf
import sys
import os
from absl import app
from absl import flags
import time
import datetime
from Net.ModelZoo.AutoEncoderMSE import AutoEncoderMSE
from Net.ModelZoo.ConcatenationEncoderDecoderMSE import ConcatenationEncoderDecoderMSE
from Net.ModelZoo.TransformationEncoderDecoderMSE import TransformationEncoderDecoderMSE
from Net.ModelZoo.TransformationEncoderMSEAE import TransformationEncoderMSEAE
from DataUtils.DataGenerator import DataGenerator

# set up flags
FLAGS = flags.FLAGS

flags.DEFINE_string('input', 'Data/train', 'Input Directory')
flags.DEFINE_string('ckpt', 'Data/Checkpoints', 'Directory to store checkpoints')
flags.DEFINE_string('logs', 'Data/Logs', 'Directory to log metrics')
flags.DEFINE_string('stats', 'Data/stats', 'Directory to store stats')
flags.DEFINE_boolean('resume_training', False, "Resume training")
flags.DEFINE_boolean('standardize_data', True, 'standardize data before training')

flags.DEFINE_string('model_code', 'TEDM', 'Defines the model to load')
flags.DEFINE_integer('output_dims', 57, 'Number of key points in output')
flags.DEFINE_integer('enc_size', 512, 'Hidden units in Encoder RNN')
flags.DEFINE_integer('dec_size', 512, 'Hidden units in Encoder RNN')
flags.DEFINE_integer('enc_layers', 1, 'Number of layers in encoder')
flags.DEFINE_integer('dec_layers', 1, 'Number of layers in decoder')
flags.DEFINE_float('enc_drop', 0.2, 'Encoder dropout probability')
flags.DEFINE_float('dec_drop', 0.2, 'SequenceDecoder dropout probability')
flags.DEFINE_float('disc_drop', 0.2, 'SequenceDiscriminator dropout probability')

flags.DEFINE_integer('epochs', 100, 'Number of training epochs')
flags.DEFINE_integer('buffer_size', 5000, 'shuffle buffer size')
flags.DEFINE_integer('batch_size', 64, 'Mini batch size')
flags.DEFINE_integer('save', 10, 'Checkpoint save epochs')
flags.DEFINE_float('learning_rate', 0.0001, 'learning rate')


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

    elif FLAGS.model_code in ["CEDM"]:
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


def combine_losses(loss1, loss2):
    """
    Combines the losses from 2 dicts and averages them
    :param loss1: loss dictionary 1
    :param loss2: loss dictionary 2
    :return: loss dict avergaing two losses
    """

    # make a copy of the losses
    loss = dict(loss1)
    if loss2 == {}:
        for k in loss1.keys():
            for loss_t in loss1[k].keys():
                loss[k][loss_t] = loss1[k][loss_t]
    else:
        for k in loss1.keys():
            for loss_t in loss1[k].keys():
                loss[k][loss_t] = loss1[k][loss_t] + loss2[k][loss_t]

    return loss


def get_model():
    """
    Returns the appropriate model
    :return:
    """

    if FLAGS.model_code == "AE":
        return AutoEncoderMSE(
            FLAGS.enc_size,
            FLAGS.batch_size,
            FLAGS.enc_layers,
            FLAGS.enc_drop,
            FLAGS.dec_size,
            FLAGS.output_dims,
            FLAGS.dec_layers,
            FLAGS.dec_drop,
            FLAGS.learning_rate
        )

    elif FLAGS.model_code == "CEDM":
        return ConcatenationEncoderDecoderMSE(
            FLAGS.enc_size,
            FLAGS.batch_size,
            FLAGS.enc_layers,
            FLAGS.enc_drop,
            FLAGS.dec_size,
            FLAGS.output_dims,
            FLAGS.dec_layers,
            FLAGS.dec_drop,
            FLAGS.learning_rate,
            os.path.join(FLAGS.ckpt, 'AE')
        )

    elif FLAGS.model_code == "TEDM":
        return TransformationEncoderDecoderMSE(
            FLAGS.enc_size,
            FLAGS.batch_size,
            FLAGS.enc_layers,
            FLAGS.enc_drop,
            FLAGS.dec_size,
            FLAGS.output_dims,
            FLAGS.dec_layers,
            FLAGS.dec_drop,
            FLAGS.learning_rate,
            os.path.join(FLAGS.ckpt, 'AE')
        )

    elif FLAGS.model_code == "TEMA":
        return TransformationEncoderMSEAE(
            FLAGS.enc_size,
            FLAGS.batch_size,
            FLAGS.enc_layers,
            FLAGS.enc_drop,
            FLAGS.dec_size,
            FLAGS.output_dims,
            FLAGS.dec_layers,
            FLAGS.dec_drop,
            FLAGS.learning_rate,
            os.path.join(FLAGS.ckpt, 'AE')
        )


def publish_logs(train_log_dir, epoch_loss, epoch):
    """
    Publishes the training logs
    :param train_log_dir: Log directory for the run
    :param epoch_loss: loss dictionary for the epoch
    :param epoch: epoch number
    :return:
    """
    for k in epoch_loss.keys():
        for loss, value in epoch_loss[k].items():
            write_dir = os.path.join(train_log_dir, k)
            train_summary_writer = tf.summary.create_file_writer(write_dir)
            with train_summary_writer.as_default():
                tf.summary.scalar(loss, data=value.numpy(), step=epoch)


def main(args):
    # load the required model
    model = get_model()

    # load the model to resume training
    if FLAGS.resume_training:
        try:
            model.load_model(
                os.path.join(FLAGS.ckpt, FLAGS.model_code)
            )
        except:
            print("Error loading model from checkpoint")
            sys.exit()

    # get the data generator object
    data_generator = DataGenerator(
        FLAGS.standardize_data,
        FLAGS.stats
    )

    # prepare the dataset
    dataset = data_generator.prepare_dataset(
        FLAGS.input,
        FLAGS.buffer_size,
        FLAGS.batch_size,
        True
    )

    # calculate the number of steps per each epoch
    steps_per_epoch = dataset.reduce(0, lambda x, _: x + 1).numpy()

    # create the tf summary writer
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(os.path.join(FLAGS.logs, current_time))

    for epoch in range(FLAGS.epochs):

        # define the epoch_loss and batch
        epoch_loss = {}
        batch = 0
        start = time.time()

        # start Mini Batch Gradient Descent
        for b, l, r in dataset.shuffle(FLAGS.buffer_size).take(steps_per_epoch):
            # discard the trajectory start states
            _, b = b
            _, l = l
            _, r = r

            # create input and target sequences
            input_seq1, target_seq1, input_seq2, target_seq2 = get_input_target(b, l, r)

            # run the training
            loss1 = model.train_step(input_seq1, target_seq1)
            loss2 = model.train_step(input_seq2, target_seq2)

            # get the batch_loss and epoch_loss
            batch_loss = combine_losses(loss1, loss2)
            epoch_loss = combine_losses(batch_loss, epoch_loss)

            # print the batch stats
            if batch % 10 == 0:
                progress = "Total Epoch {} Batch {}".format(epoch + 1, batch)
                print(progress)
                for k in batch_loss.keys():
                    for loss, value in batch_loss[k].items():
                        loss_string = "\t{} Loss {}".format(loss, value.numpy())
                        print(loss_string)

            # update the batch number
            batch += 1

        # save checkpoints
        if (epoch + 1) % FLAGS.save == 0:
            try:
                model.save_model(
                    os.path.join(FLAGS.ckpt, FLAGS.model_code)
                )
            except:
                print("Error saving model checkpoint")
                sys.exit()

        # print progress
        progress = "\tTotal Epoch {}".format(epoch + 1)
        print(progress)
        for k in epoch_loss.keys():
            for loss, value in epoch_loss[k].items():
                loss_string = "\t\t{} Loss {}".format(loss, value.numpy())
                print(loss_string)

        print("\tEpoch train time {}".format(time.time() - start))

        publish_logs(train_log_dir, epoch_loss, epoch)


if __name__ == '__main__':
    app.run(main)
