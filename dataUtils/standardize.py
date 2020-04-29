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

class StandardizeClass:

    def __init__(self):
        self.mean_data = np.loadtxt(FLAGS.meanCsv, delimiter=",")
        self.std_data = np.loadtxt(FLAGS.stdCsv, delimiter=",")
        self.mean_data = np.expand_dims(self.mean_data, axis=0)
        self.std_data = np.expand_dims(self.std_data, axis=0)
        self.mean_data_tensor = tf.convert_to_tensor(self.mean_data, dtype=tf.float32)
        self.std_data_tensor = tf.convert_to_tensor(self.std_data, dtype=tf.float32)

    @staticmethod
    def deserialize_example(example):

        buyerJoints = tf.cast   (tf.io.parse_tensor(example['br'], out_type=tf.double), tf.float32)
        leftSellerJoints = tf.cast(tf.io.parse_tensor(example['ls'], out_type=tf.double), tf.float32)
        rightSellerJoints = tf.cast(tf.io.parse_tensor(example['rs'], out_type=tf.double), tf.float32)

        return buyerJoints, leftSellerJoints, rightSellerJoints

    def standardize(self, joints):

        joints = joints - self.mean_data_tensor

        joints = tf.divide(joints, self.std_data_tensor)

        return joints

    def destandardize(self, joints):

        joints = np.multiply(joints, self.std_data) + self.mean_data

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
    dataset = dataset.map(StandardizeClass.deserialize_example)

    merged = []

    for b, l, r in dataset:
        res = np.vstack((b.numpy(), l.numpy(), r.numpy()))
        merged.append(res)

    merged = np.vstack(merged)

    mean_data = np.mean(merged, axis=0)
    std_data = np.std(merged, axis=0)

    mean_data = np.expand_dims(mean_data, axis=0)
    std_data = np.expand_dims(std_data, axis=0)

    np.savetxt(FLAGS.meanCsv, mean_data, delimiter=",")
    np.savetxt(FLAGS.stdCsv, std_data, delimiter=",")
            
if __name__ == '__main__':
    app.run(main)
