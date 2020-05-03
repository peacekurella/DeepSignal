import numpy as np
import os
import sys
import tensorflow as tf

tf.executing_eagerly()


class InputStandardizer:

    def __init__(self, stats_dir):
        """
        Constructor for the InputStandardizer class
        :param stats_dir:
        """

        mean_file = os.path.join(stats_dir, 'mean.csv')
        std_file = os.path.join(stats_dir, 'std.csv')

        self.mean = np.loadtxt(mean_file, delimiter=",")
        self.mean = np.expand_dims(self.mean, axis=0)
        self.mean = tf.convert_to_tensor(self.mean, dtype=tf.float32)

        self.std = np.loadtxt(std_file, delimiter=",")
        self.std = np.expand_dims(self.std, axis=0)
        self.std = tf.convert_to_tensor(self.std, dtype=tf.float32)

    def standardize(self, joints):
        """
        Standardizes the data
        :param joints: input original trajectory
        :return: standardized trajectory
        """

        joints = joints - self.mean
        joints = tf.divide(joints, self.std)

        return joints

    def destandardize(self, joints):
        """
        destandardizes the input 
        :param joints: input standardized trajectory
        :return: destandardized trajectory
        """
        joints = np.multiply(joints, self.std.numpy()) + self.mean.numpy()

        return joints


def deserialize_example(example):
    """
    deserializes a serialized example
    :return: tuple of buyer, left and right seller joints
    """

    buyerJoints = tf.cast(tf.io.parse_tensor(example['br'], out_type=tf.double), tf.float32)
    leftSellerJoints = tf.cast(tf.io.parse_tensor(example['ls'], out_type=tf.double), tf.float32)
    rightSellerJoints = tf.cast(tf.io.parse_tensor(example['rs'], out_type=tf.double), tf.float32)

    return buyerJoints, leftSellerJoints, rightSellerJoints


def parse_example(example_proto):
    """
    Parses examples from the record files
    :param example_proto: input example proto_buffer string
    :return: parsed example
    """

    # create a feature descriptor
    feature_description = {
        'br': tf.io.FixedLenFeature([], tf.string),
        'b_start': tf.io.FixedLenFeature([], tf.string),
        'ls': tf.io.FixedLenFeature([], tf.string),
        'l_start': tf.io.FixedLenFeature([], tf.string),
        'rs': tf.io.FixedLenFeature([], tf.string),
        'r_start': tf.io.FixedLenFeature([], tf.string),

    }

    return tf.io.parse_single_example(example_proto, feature_description)


def generate_stats(train_dir, stats_dir):
    """
    Generates column wise mean, std from train data
    :param train_dir: directory contain train TFrecords
    :param stats_dir: directory to write stats in
    :return: 
    """

    if not (os.path.isdir(train_dir)):
        print("Invalid directory for train data")
        sys.exit()

    if not (os.path.isdir(stats_dir)):
        try:
            os.mkdir(stats_dir)
        except:
            print("Error creating stats directory")
            sys.exit()

    # set the directories
    mean_file = os.path.join(stats_dir, 'mean.csv')
    std_file = os.path.join(stats_dir, 'std.csv')

    # read the files
    files = list(map(lambda x: os.path.join(train_dir, x), os.listdir(train_dir)))
    dataset = tf.data.TFRecordDataset(files)

    # parse the examples
    dataset = dataset.map(parse_example)

    # deserialize the tensors
    dataset = dataset.map(deserialize_example)

    merged = []
    for b, l, r in dataset:
        res = np.vstack((b.numpy(), l.numpy(), r.numpy()))
        merged.append(res)

    merged = np.vstack(merged)

    mean = np.mean(merged, axis=0)
    std = np.std(merged, axis=0)

    mean = np.expand_dims(mean, axis=0)
    std = np.expand_dims(std, axis=0)

    np.savetxt(mean_file, mean, delimiter=",")
    np.savetxt(std_file, std, delimiter=",")
