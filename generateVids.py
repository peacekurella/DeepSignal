import pickle
import os
from dataVis import create_animation
import sys
from absl import app
from absl import flags

# set up flags
FLAGS = flags.FLAGS

flags.DEFINE_string('input', 'input', 'Input Directory')
flags.DEFINE_string('output', 'output', 'Output Directory')

def main(argv):
    """
    Generates videos from raw data set
    :param argv: Input arguments
    :return: None
    """

    if not (os.path.isdir(FLAGS.input)):
        print("Invalid input directory")
        sys.exit()

    if not (os.path.isdir(FLAGS.output)):
        try:
            os.mkdir(FLAGS.output)
        except:
            print("Error creating output directory")
            sys.exit()

    for file in os.listdir(FLAGS.input):

        # load the data
        pkl = os.path.join(FLAGS.input, file)
        pkl = open(pkl, 'rb')
        group = pickle.load(pkl)

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
            if(subject['humanId'] == leftSellerId):
                leftSellerJoints = subject['joints19']
            if (subject['humanId'] == rightSellerId):
                rightSellerJoints = subject['joints19']
            if (subject['humanId'] == buyerId):
                buyerJoints = subject['joints19']

        # create a new video only if it doesn't exist
        if not (os.path.isfile(os.path.join(FLAGS.output, file.split('.')[0]+'.mp4'))):
            create_animation(buyerJoints, leftSellerJoints, rightSellerJoints, os.path.join(FLAGS.output, file.split('.')[0]))

if __name__== '__main__':
    app.run(main)


