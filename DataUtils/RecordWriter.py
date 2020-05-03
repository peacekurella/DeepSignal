import pickle
import tensorflow as tf
import os
import sys
import random
from .TrajectoryHandler import convert_to_trajectory

tf.executing_eagerly()


def _tensor_feature(value):
    """
    returns a bytes_list from a tensor
    :param value: Tensor to be converted to bytes_list
    :return: bytes_list form of the input tensor
    """
    value = tf.io.serialize_tensor(value)
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_example(b_start, buyerJoints, l_start, leftSellerJoints, r_start, rightSellerJoints):
    """
    Serializes a training example
    :param r_start: start position of right_seller
    :param l_start: start position of left seller
    :param b_start: start position of buyer
    :param buyerJoints: buyer joint Sequence
    :param leftSellerJoints:  left seller joint joint sequence
    :param rightSellerJoints: right seller joint sequence
    :return: serialized dict of features
    """

    # create a dict of features
    feature = {
        'br': _tensor_feature(buyerJoints),
        'b_start': _tensor_feature(b_start),
        'ls': _tensor_feature(leftSellerJoints),
        'l_start': _tensor_feature(l_start),
        'rs': _tensor_feature(rightSellerJoints),
        'r_start': _tensor_feature(r_start)
    }

    # serialize feature dict
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def tf_serialize_example(b_start, buyerJoints, l_start, leftSellerJoints, r_start, rightSellerJoints):
    """
    Wrapper function for serialize example
    :param buyerJoints: buyer joint Sequence
    :param leftSellerJoints:  left seller joint joint sequence
    :param rightSellerJoints: right seller joint sequence
    :param b_start: start position of buyer
    :param l_start:  start position of left seller
    :param r_start: start_position of right_seller
    :return: serialized tensor string
    """
    args = (b_start, buyerJoints, l_start, leftSellerJoints, r_start, rightSellerJoints)
    tf_string = tf.py_function(serialize_example, args, tf.string)
    return tf.reshape(tf_string, ())


# noinspection PyBroadException
class RecordWriter:
    """
    Handles the creation of TF record files from pkl files
    """

    def __init__(self, input_dir, train_dir, test_dir, max_test_files, key_points, seq_len):
        """
        Defines the constructor for the RecordWriter class
        :param seq_len: sequence length to be generated
        :param input_dir: Input pickle file directory
        :param train_dir: Directory to put train records in
        :param test_dir: Directory to put test records in   
        :param max_test_files: Maximum number of test files
        :param key_points: Number of key points in the skeleton
        """

        self.input_dir = input_dir
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.max_test_files = max_test_files
        self.keypoints = key_points
        self.seqLength = seq_len

    def split_into_features(self, sequence):
        """
        Mapper function to split sequence into features
        :param sequence: Tensor of shape (seqLength, 3, keypoints, 1)
        :return: Tuple of Tensors of shape (seqLength, keypoints)
        """

        # set sequence length
        seqLength = sequence.shape[0]

        # split into features
        buyerJoints, leftSellerJoints, rightSellerJoints = tf.split(sequence, 3, axis=1)

        # preserve order for proper visualization
        buyerJoints = tf.reshape(buyerJoints, (seqLength, self.keypoints))
        rightSellerJoints = tf.reshape(rightSellerJoints, (seqLength, self.keypoints))
        leftSellerJoints = tf.reshape(leftSellerJoints, (seqLength, self.keypoints))

        # convert to trajectories
        b_start, buyerJoints = convert_to_trajectory(buyerJoints)
        l_start, leftSellerJoints = convert_to_trajectory(leftSellerJoints)
        r_start, rightSellerJoints = convert_to_trajectory(rightSellerJoints)

        return b_start, buyerJoints, l_start, leftSellerJoints, r_start, rightSellerJoints

    def generate_dataset(self, buyerJoints, leftSellerJoints, rightSellerJoints, seqLength):
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

        # flatten dataset coming from window
        dataset = dataset.flat_map(lambda x: x)
        dataset = dataset.batch(seqLength, drop_remainder=True)

        # generate feature splits
        dataset = dataset.map(self.split_into_features)
        return dataset

    def create_records(self):
        """
        Creates the TF record files
        :return: None
        """
        if not (os.path.isdir(self.input_dir)):
            print("Invalid input directory")
            sys.exit()

        if not (os.path.isdir(self.train_dir)):
            try:
                os.mkdir(self.train_dir)
            except:
                print("Error creating train directory")
                sys.exit()

        if not (os.path.isdir(self.test_dir)):
            try:
                os.mkdir(self.test_dir)
            except:
                print("Error creating test directory")
                sys.exit()

        # randomly shuffle the files
        files = os.listdir(self.input_dir)
        random.shuffle(files)

        for count, file in enumerate(files):

            # load the data
            pkl = os.path.join(self.input_dir, file)
            pkl = open(pkl, 'rb')
            group = pickle.load(pkl)
            seqLength = self.seqLength

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
                if subject['humanId'] == leftSellerId:
                    leftSellerJoints = subject['joints19']
                if subject['humanId'] == rightSellerId:
                    rightSellerJoints = subject['joints19']
                if subject['humanId'] == buyerId:
                    buyerJoints = subject['joints19']

            if len(buyerJoints[0]) > self.seqLength:
                # generate the dataset
                dataset = self.generate_dataset(buyerJoints, leftSellerJoints, rightSellerJoints, seqLength)
                record = dataset.map(tf_serialize_example)

                # write to TFrecord file
                if file.find("a1") >= 0:
                    filename = os.path.join(self.test_dir, file.split('.')[0] + ".TFrecord")
                else:
                    filename = os.path.join(self.train_dir, file.split('.')[0] + ".TFrecord")
                writer = tf.data.experimental.TFRecordWriter(filename)
                writer.write(record)
                print(filename)
