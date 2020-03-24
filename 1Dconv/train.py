import sys
sys.path.append("..")
import modelBuilder as modelBuilder
import tensorflow as tf
from absl import flags
from absl import app
from dataUtils import getData as db
import os
import time

# set up flags
FLAGS = flags.FLAGS

flags.DEFINE_string('input', '../train', 'Input Directory')
flags.DEFINE_string('ckpt', '../ckpt', 'Directory to store checkpoints')
flags.DEFINE_string('logs', '../logs', 'Directory to log metrics')

flags.DEFINE_integer('epochs', 200, 'Number of training epochs')
flags.DEFINE_float('dropout', 0.25, 'dropout probability')
flags.DEFINE_integer('keypoints', 57, 'Number of keypoints')
flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
flags.DEFINE_integer('buffer_size', 5000, 'shuffle buffer size')
flags.DEFINE_integer('batch_size', 64, 'Mini batch size')
flags.DEFINE_integer('save', 100, 'Checkpoint save epochs')
flags.DEFINE_integer('seqLength', 116, 'sequence to be passed for predictions')
flags.DEFINE_boolean("load_ckpt", False, 'Resume training from loaded checkpoint')
flags.DEFINE_boolean("train_pe", False, 'Train the pose encoder')


@tf.function
def train_step_pe(input, poseEncoder, motionDecoder, optimizer):

    # initialize loss
    loss = 0

    # calculate the loss
    with tf.GradientTape() as tape:
        encoder_out = poseEncoder(input, training=True)
        predictions = motionDecoder(encoder_out, training=True)
        loss = tf.keras.losses.MSE(input, predictions)
        loss = tf.reduce_mean(loss)

    # get trainable variables
    variables = poseEncoder.trainable_variables + motionDecoder.trainable_variables

    # get the gradients
    gradients = tape.gradient(loss, variables)

    # purge from memory
    del tape, predictions, encoder_out

    # apply gradients to variables
    optimizer.apply_gradients(zip(gradients, variables))

    # purge from memory
    del optimizer, gradients, variables

    return loss


@tf.function
def train_step(input, target, motionEncoder, motionDecoder, optimizer):

    # initialize loss
    loss = 0

    # calculate the loss
    with tf.GradientTape() as tape:

        encoder_out = motionEncoder(input, training=True)
        predictions = motionDecoder(encoder_out, training=True)
        loss = tf.keras.losses.MSE(target, predictions)
        loss = tf.reduce_mean(loss)

    # get trainable variables
    variables = motionEncoder.trainable_variables

    # get the gradients
    gradients = tape.gradient(loss, variables)

    # purge from memory
    del tape, predictions, encoder_out

    # apply gradients to variables
    optimizer.apply_gradients(zip(gradients, variables))

    # purge from memory
    optimizer, gradients, variables

    return loss

def main(args):

    # prepare the dataset
    input = FLAGS.input
    buffer_size = FLAGS.buffer_size
    batch_size = FLAGS.batch_size
    dataset = db.prepare_dataset(input, buffer_size, batch_size, True)

    # set up experiment
    steps_per_epoch = dataset.reduce(0, lambda x, _: x + 1).numpy()
    dropout_rate = FLAGS.dropout
    keypoints = FLAGS.keypoints
    learning_rate = FLAGS.learning_rate
    logs = FLAGS.logs
    epochs = FLAGS.epochs
    save = FLAGS.save
    seqLength = FLAGS.seqLength
    load_ckpt = FLAGS.load_ckpt
    train_pe = FLAGS.train_pe
    count_ckpt = 0
    print(train_pe)

    # set up the model
    poseEncoder = modelBuilder.poseEncoder(dropout_rate)
    motionEncoder = modelBuilder.motionEncoder(dropout_rate)
    motionDecoder = modelBuilder.motionDecoder(keypoints)

    # set up optimizer
    optimizer = tf.keras.optimizers.Adam(
        lr=learning_rate
    )

    # create checkpoint saver
    checkpoint = tf.train.Checkpoint(
        optimizer=optimizer,
        poseEncoder = poseEncoder,
        motionEncoder = motionEncoder,
        motionDecoder = motionDecoder
    )

    # load checkpoint if needed
    if load_ckpt:
        checkpoint.restore(
            tf.train.latest_checkpoint(FLAGS.ckpt)
        )

    # set up summary writers
    train_log_dir = os.path.join(os.path.join(logs, 'conv1d'),  'train')
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    if train_pe:
        print("Training PE")
        # start training the pose auto encoder
        for epoch in range(epochs):

            # measure start time
            start = time.time()
            epoch_loss = 0

            # do the actual training
            for (batch, (b, l, r)) in enumerate(dataset.shuffle(batch_size).take(steps_per_epoch)):

                # slice to sequence length
                l = l[:, :seqLength, :]

                # train with left seller body
                batch_loss = train_step_pe(l, poseEncoder, motionDecoder, optimizer)

                # log the loss
                with train_summary_writer.as_default():
                    tf.summary.scalar('Reconstruction_Error', data=batch_loss.numpy(), step=epoch)

                epoch_loss += batch_loss

                # print progress periodically
                # save logs for tensorboard
                if batch % 100 == 0:
                    progress = "Epoch {} Batch {} Loss {:.4f}".format(epoch + 1, batch, batch_loss.numpy())
                    print(progress)

            # save checkpoint
            if (epoch + 1) % save == 0:
                ckpt_prefix = os.path.join(FLAGS.ckpt, "ckpt{}".format(count_ckpt))
                checkpoint.save(ckpt_prefix)
                count_ckpt += 1

            # print progress
            progress = "\tAvg Epoch {} Loss {:.4f}".format(epoch + 1, (epoch_loss / steps_per_epoch))
            print(progress)
            print("\tEpoch train time {}".format(time.time() - start))

    # start training the motion encoder
    for epoch in range(epochs):

        # measure start time
        start = time.time()
        epoch_loss = 0

        # do the actual training
        for (batch, (b, l, r)) in enumerate(dataset.shuffle(batch_size).take(steps_per_epoch)):

            # input and target sequences
            input = tf.concat([b[:, :seqLength, :], r[:, :seqLength, :]], axis=2)
            target = l[:, :seqLength, :]

            # actual train step
            batch_loss = train_step(input, target, motionEncoder, motionDecoder, optimizer)

            # log the loss
            with train_summary_writer.as_default():
                tf.summary.scalar('MSE', data=batch_loss.numpy(), step=epoch)

            epoch_loss += batch_loss

            # print progress periodically
            # save logs for tensorboard
            if batch % 100 == 0:
                progress = "Epoch {} Batch {} Loss {:.4f}".format(epoch + 1, batch, batch_loss.numpy())
                print(progress)

        # save checkpoint
        if (epoch + 1) % save == 0:
            ckpt_prefix = os.path.join(FLAGS.ckpt, "ckpt{}".format(count_ckpt))
            checkpoint.save(ckpt_prefix)
            count_ckpt += 1

        # print progress
        progress = "\tAvg Epoch {} Loss {:.4f}".format(epoch + 1, (epoch_loss / steps_per_epoch))
        print(progress)
        print("\tEpoch train time {}".format(time.time() - start))

if __name__ == '__main__':
    app.run(main)
