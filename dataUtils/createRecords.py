import pickle
import tensorflow as tf
import os
from absl import app
from absl import flags
import sys
import random

# set up flags
FLAGS = flags.FLAGS

flags.DEFINE_string('input', '../normalized', 'Input Directory')
flags.DEFINE_string('train', '../train', 'Train record output directory')
flags.DEFINE_string('test', '../test', 'Train record output directory')
flags.DEFINE_integer('seqLength', 57, 'Sequence length')
flags.DEFINE_integer('testFiles', 50, 'Number of pkl files used for testing')
flags.DEFINE_integer('keypoints', 57, 'Number of keypoints in the skeleton')


def split_into_features(sequence):
    """
    Mapper function to split sequence into features
    :param sequence: Tensor of shape (seqLength, 3, keypoints, 1)
    :return: Tuple of Tensors of shape (seqLength, keypoints)
    """

    # set sequence length
    seqLength = sequence.shape[0]

    # split into features
    buyerJoints, rightSellerJoints, leftSellerJoints = tf.split(sequence, 3, axis=1)

    # preserve order for proper visualization
    buyerJoints = tf.reshape(buyerJoints, (seqLength, FLAGS.keypoints))
    rightSellerJoints = tf.reshape(rightSellerJoints, (seqLength, FLAGS.keypoints))
    leftSellerJoints = tf.reshape(leftSellerJoints, (seqLength, FLAGS.keypoints))
    return buyerJoints, rightSellerJoints, leftSellerJoints

def generate_dataset(buyerJoints, leftSellerJoints, rightSellerJoints, seqLength):
    """
    creates a dataset object with features as tuples
    :param buyerJoints: Sequence of buyerJoints
    :param leftSellerJoints: Sequence of leftSellerJoints
    :param rightSellerJoints: Sequence of rightSellerJoints
    :param seqLength: length of sequences
    :return: Tensorflow dataset object with feature tuples
    """
    # stack all features
    sequence = tf.stack([buyerJoints, leftSellerJoints, rightSellerJoints])

    # split by time axis to create a tensor
    sequence = tf.stack(tf.split(sequence, len(buyerJoints[0]), axis=2))

    # create continuous sequences of seqLength
    dataset = tf.data.Dataset.from_tensor_slices(sequence)
    dataset = dataset.window(seqLength, 2, 1, True)

    # flatten dataset comin from window
    dataset = dataset.flat_map(lambda x: x)
    dataset = dataset.batch(seqLength, drop_remainder=True)

    # generate feature splits
    dataset = dataset.map(split_into_features)
    return dataset

def _tensor_feature(value):
    """
    returns a bytes_list from a tensor
    :param value: Tensor to be converted to bytes_list
    :return: bytes_list form of the input tensor
    """
    value = tf.io.serialize_tensor(value)
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_example(buyerJoints, leftSellerJoints, rightSellerJoints):
    """
    Serializes a training example
    :param buyerJoints: buyer joint Sequence
    :param leftSellerJoints:  left seller joint joint sequence
    :param rightSellerJoints: right seller joint sequence
    :return: serialized dict of features
    """

    # create a dict of features
    feature = {
        'br': _tensor_feature(buyerJoints),
        'ls': _tensor_feature(leftSellerJoints),
        'rs': _tensor_feature(rightSellerJoints)
    }

    # serialize feature dict
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def tf_serialize_example(buyerJoints, leftSellerJoints, rightSellerJoints):
    """
    Wrapper function for serialize_example
    :param buyerJoints: buyer joint Sequence
    :param leftSellerJoints:  left seller joint joint sequence
    :param rightSellerJoints: right seller joint sequence
    :return: Serialized features
    """
    args = (buyerJoints, leftSellerJoints, rightSellerJoints)
    tf_string = tf.py_function(serialize_example, args, tf.string)
    return tf.reshape(tf_string, ())

def main(argv):
    """
    Generates videos from raw data set
    :param argv: Input arguments
    :return: None
    """

    if not (os.path.isdir(FLAGS.input)):
        print("Invalid input directory")
        sys.exit()

    if not (os.path.isdir(FLAGS.train)):
        try:
            os.mkdir(FLAGS.train)
        except:
            print("Error creating train directory")
            sys.exit()

    if not (os.path.isdir(FLAGS.test)):
        try:
            os.mkdir(FLAGS.test)
        except:
            print("Error creating test directory")
            sys.exit()

    # randomly shuffle the files
    files = os.listdir(FLAGS.input)
    random.shuffle(files)

    for count, file in enumerate(files):

        # load the data
        pkl = os.path.join(FLAGS.input, file)
        pkl = open(pkl, 'rb')
        group = pickle.load(pkl)
        seqLength = FLAGS.seqLength

        # load buyer, seller Ids
        leftSellerId = group['leftSellerId']
        rightSellerId = group['rightSellerId']
        buyerId = group['buyerId']

        # load skeletons
        buyerJoints = []
        leftSellerJoints = []
        rightSellerJoints = []

        # load the skeletons
        for subject in group['subjects']:
            if (subject['humanId'] == leftSellerId):
                leftSellerJoints = subject['joints19']
            if (subject['humanId'] == rightSellerId):
                rightSellerJoints = subject['joints19']
            if (subject['humanId'] == buyerId):
                buyerJoints = subject['joints19']

        # create dataset from the sequences
        dataset = generate_dataset(buyerJoints, leftSellerJoints, rightSellerJoints, seqLength)
        record = dataset.map(tf_serialize_example)

        # write to TFrecord file
        if(count < FLAGS.testFiles):
            filename = os.path.join(FLAGS.test, file.split('.')[0]+'.TFrecord')
        else:
            filename = os.path.join(FLAGS.train, file.split('.')[0] + '.TFrecord')
        writer = tf.data.experimental.TFRecordWriter(filename)
        writer.write(record)
        print(filename)

if __name__ == '__main__':
    app.run(main)




