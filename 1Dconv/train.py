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

flags.DEFINE_integer('epochs', 1000, 'Number of training epochs')
flags.DEFINE_float('dropout', 0.25, 'dropout probability')
flags.DEFINE_integer('keypoints', 57, 'Number of keypoints')
flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
flags.DEFINE_integer('buffer_size', 5000, 'shuffle buffer size')
flags.DEFINE_integer('batch_size', 64, 'Mini batch size')
flags.DEFINE_integer('save', 100, 'Checkpoint save epochs')
flags.DEFINE_integer('seqLength', 56, 'sequence to be passed for predictions')
flags.DEFINE_boolean("load_ckpt", False, 'Resume training from loaded checkpoint')

@tf.function
def train_step(input, target, model, optimizer):

    # calculate the loss
    with tf.GradientTape() as tape:

        predictions = model(input, training=True)
        loss = tf.keras.losses.MSE(target, predictions)
        loss = tf.reduce_mean(loss)

    # get trainable variables
    variables = model.trainable_variables

    # get the gradients
    gradients = tape.gradient(loss, variables)

    # purge tape from memory
    del tape

    # apply gradients to variables
    optimizer.apply_gradients(zip(gradients, variables))

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

    # set up the model
    model = modelBuilder.motionAutoEncoder(dropout_rate, keypoints)

    # set up optimizer
    optimizer = tf.keras.optimizers.Adam(
        lr=learning_rate
    )

    # create checkpoint saver
    checkpoint = tf.train.Checkpoint(
        optimizer=optimizer,
        model=model
    )

    # load checkpoint if needed
    if load_ckpt:
        checkpoint.restore(
            tf.train.latest_checkpoint(FLAGS.ckpt)
        )

    # set up summary writers
    train_log_dir = os.path.join(os.path.join(logs, 'conv1d'),  'train')
    profiler_log_dir = os.path.join(os.path.join(logs, 'conv1d'),  'func')
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    profiler_summary_writer = tf.summary.create_file_writer(profiler_log_dir)

    # start training
    for epoch in range(epochs):

        # measure start time
        start = time.time()
        epoch_loss = 0

        # do the actual training
        for (batch, (b, l, r)) in enumerate(dataset.shuffle(batch_size).take(steps_per_epoch)):

            # input and target sequences
            input = tf.concat([b[:, :seqLength, :], r[:, :seqLength, :]], axis=2)
            target = l[:, :seqLength, :]

            # do the train step
            if (epoch == 0 and batch == 0):
                tf.summary.trace_on(graph=True, profiler=False)

            # actual train step
            batch_loss = train_step(input, target, model, optimizer)

            # log the trace
            if (epoch == 0 and batch == 0):
                with profiler_summary_writer.as_default():
                    tf.summary.trace_export(
                        name="Execution Graph",
                        step=0
                    )

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
            ckpt_prefix = os.path.join(FLAGS.ckpt, "ckpt")
            checkpoint.save(ckpt_prefix)

        # print progress
        progress = "\tAvg Epoch {} Loss {:.4f}".format(epoch + 1, (epoch_loss / steps_per_epoch))
        print(progress)
        print("\tEpoch train time {}".format(time.time() - start))

if __name__ == '__main__':
    app.run(main)
