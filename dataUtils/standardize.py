import csv
import numpy as np
import os
import sys
import tensorflow as tf
from absl import flags
from absl import app
from pathlib import Path

sys.path.append('..')
import dataUtils.getData as gd

# set up flags
FLAGS = flags.FLAGS

flags.DEFINE_string('trainData', '../train', 'Train record directory')
flags.DEFINE_string('meanCsv', 'mean.csv', 'CSV file to store mean')
flags.DEFINE_string('stdCsv', 'std.csv', 'CSV file to store standard deviation')

def deserialize_example(example):
    buyerJoints = tf.cast(tf.io.parse_tensor(example['br'], out_type=tf.double), tf.float32)
    leftSellerJoints = tf.cast(tf.io.parse_tensor(example['ls'], out_type=tf.double), tf.float32)
    rightSellerJoints = tf.cast(tf.io.parse_tensor(example['rs'], out_type=tf.double), tf.float32)

    return buyerJoints, leftSellerJoints, rightSellerJoints

def standardize(joints):

    mean_data = np.loadtxt("mean.csv", delimiter=",")
    std_data = np.loadtxt("std.csv", delimiter=",")

    mean_data = mean_data.reshape(57, 1)
    mean_data = np.transpose(mean_data)
    std_data = std_data.reshape(57, 1)
    std_data = np.transpose(std_data)

    mean_data = tf.convert_to_tensor(mean_data, dtype=tf.float32)
    std_data = tf.convert_to_tensor(std_data, dtype=tf.float32)

    joints = joints - mean_data

    joints = tf.divide(joints, std_data)

    return joints

def destandardize(joints):

    mean_data = np.loadtxt("mean.csv", delimiter=",")
    std_data = np.loadtxt("std.csv", delimiter=",")

    mean_data = mean_data.reshape(57, 1)
    mean_data = np.transpose(mean_data)
    std_data = std_data.reshape(57, 1)
    std_data = np.transpose(std_data)

    joints = np.multiply(joints, std_data) + mean_data

    return joints

def main(argv):
    """
    Generates normalized tf records form the train data
    :param argv: Input arguments
    :return: None
    """

    if not (os.path.isdir(FLAGS.trainData)):
        print("Invalid directory for train data")
        sys.exit()                                                                 

    tf.compat.v1.enable_eager_execution()
    # read the files
    files = list(map(lambda x: os.path.join(FLAGS.trainData, x), os.listdir(FLAGS.trainData)))
    dataset = tf.data.TFRecordDataset(files)

    # parse the examples
    dataset = dataset.map(gd.parse_example)

    # deserialize the tensors
    dataset = dataset.map(deserialize_example)

    merged = []

    for b, l, r in dataset:
        res = np.vstack((b.numpy(), l.numpy(), r.numpy()))
        merged.append(res)

    merged = np.vstack(merged)

    mean_data = np.mean(merged, axis=0)
    std_data = np.std(merged, axis=0)

    mean_data = mean_data.reshape(57, 1)
    mean_data = np.transpose(mean_data)
    std_data = std_data.reshape(57, 1)
    std_data = np.transpose(std_data)

    np.savetxt(FLAGS.meanCsv, mean_data, delimiter=",")
    np.savetxt(FLAGS.stdCsv, std_data, delimiter=",")
            
if __name__ == '__main__':
    app.run(main)
