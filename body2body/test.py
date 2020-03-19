import tensorflow as tf
from absl import app
from absl import flags
import train
import modelBuilder as model

# set up flags
FLAGS = flags.FLAGS

flags.DEFINE_string('test_input', '../test', 'Input Directory')

def run_inference(input_seq, target_seq, encoder, decoder):
    """
    Returns the predictions given input_seq
    :param input_seq: input sequence
    :param target_seq: target sequence
    :param encoder: encoder object
    :param decoder: decoder object
    :return:
    """

    # number of time steps the
    time_steps = target_seq.shape[1]

    # initialize encoder hidden state
    enc_hidden = encoder.initialize_hidden_state(1)

    # encoder output
    enc_output, enc_hidden = encoder(input_seq, enc_hidden, False)

    # first input to decoder
    buyer = input_seq[:, -1, :decoder.output_size]
    sellers = target_seq[:, 0, decoder.output_size:]
    dec_input = tf.concat([buyer, sellers], axis=1)
    dec_hidden = enc_hidden

    # list of predictions
    predictions = []

    for t in range(0, time_steps):
        # get the predictions
        prediction, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output, True)
        predictions.append(tf.expand_dims(prediction, axis=1))

        # update inputs to decoder
        del dec_input
        buyer = prediction
        sellers = target_seq[:, t, decoder.output_size:]
        dec_input = tf.concat([buyer, sellers], axis=1)

    return tf.concat(predictions, axis=1)


def main(args):
    """
    Runs the inference example
    :param args:
    :return: None
    """

    # prep the input for inference
    dataset = train.prepare_dataset(FLAGS.test_input)
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

    # create encoder, decoder, and optimizer
    encoder = model.Encoder(enc_size, batch_size, enc_layers, enc_drop)
    decoder = model.Decoder(keypoints, dec_size, batch_size, dec_layers, dec_drop)

    # create checkpoint saver
    checkpoint = tf.train.Checkpoint(
        encoder=encoder,
        decoder=decoder
    )
    checkpoint.restore(checkpoint_dir).expect_partial()

    # run a test prediction
    for (b, l, r) in dataset.take(1):
        input = tf.expand_dims(tf.concat([b, l, r], axis=2)[0], axis=0)
        input_seq = input[:, :inp_length]
        target_seq = input[:, inp_length:]
        buyers = run_inference(input_seq, target_seq, encoder, decoder)
        print(buyers)


if __name__ == '__main__':
    app.run(main)
