import tensorflow as tf
import os
import sys
sys.path.append('..')
import dataUtils.standardize as stdrd


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


def deserialize_example_std(example):
    """
    Deserializes the tensors in parsed examples
    :param example: input example to be parsed
    :return: (buyerJoints, leftSellerJoints, rightSellerJoints) tuple containing the sequences
    """

    std = stdrd.StandardizeClass()

    # cast to float32 for better performance
    buyerJoints = tf.cast(tf.io.parse_tensor(example['br'], out_type=tf.double), tf.float32)
    leftSellerJoints = tf.cast(tf.io.parse_tensor(example['ls'], out_type=tf.double), tf.float32)
    rightSellerJoints = tf.cast(tf.io.parse_tensor(example['rs'], out_type=tf.double), tf.float32)

    # standardize the values
    buyerJoints = std.standardize(buyerJoints)
    leftSellerJoints = std.standardize(leftSellerJoints)
    rightSellerJoints = std.standardize(rightSellerJoints)

    # cast to float32 for better performance
    b_start = tf.cast(tf.io.parse_tensor(example['b_start'], out_type=tf.double), tf.float32)
    l_start = tf.cast(tf.io.parse_tensor(example['l_start'], out_type=tf.double), tf.float32)
    r_start = tf.cast(tf.io.parse_tensor(example['r_start'], out_type=tf.double), tf.float32)

    return (b_start, buyerJoints), (l_start, leftSellerJoints), (r_start, rightSellerJoints)

def deserialize_example(example):
    """
    Deserializes the tensors in parsed examples
    :param example: input example to be parsed
    :return: (buyerJoints, leftSellerJoints, rightSellerJoints) tuple containing the sequences
    """

    # cast to float32 for better performance
    buyerJoints = tf.cast(tf.io.parse_tensor(example['br'], out_type=tf.double), tf.float32)
    leftSellerJoints = tf.cast(tf.io.parse_tensor(example['ls'], out_type=tf.double), tf.float32)
    rightSellerJoints = tf.cast(tf.io.parse_tensor(example['rs'], out_type=tf.double), tf.float32)


    # cast to float32 for better performance
    b_start = tf.cast(tf.io.parse_tensor(example['b_start'], out_type=tf.double), tf.float32)
    l_start = tf.cast(tf.io.parse_tensor(example['l_start'], out_type=tf.double), tf.float32)
    r_start = tf.cast(tf.io.parse_tensor(example['r_start'], out_type=tf.double), tf.float32)

    return (b_start, buyerJoints), (l_start, leftSellerJoints), (r_start, rightSellerJoints)


def prepare_dataset(input, buffer_size, batch_size, drop_remainder, standardize):
    """
    Prepares the dataset
    :param input: input directory
    :param buffer_size: shuffle buffer size
    :param batch_size: mini batch size
    :param drop_remainder: drop remainder from the batched examples
    :param standardize: True if we want to standardize our data
    :return: dataset object with example of shape (batch_size, seqLength, input_size)
    """

    if not (os.path.isdir(input)):
        print("Invalid input directory")
        sys.exit()

    # read the files
    files = list(map(lambda x: os.path.join(input, x), os.listdir(input)))
    dataset = tf.data.TFRecordDataset(files)

    # parse the examples
    dataset = dataset.map(parse_example)

    # deserialize the tensors
    if standardize:
        dataset = dataset.map(deserialize_example_std)
    else:
        dataset = dataset.map(deserialize_example)

    # shuffle and batch the data
    dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=drop_remainder)

    return dataset
