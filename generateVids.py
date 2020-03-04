import pickle
import os
import argparse
from dataVis import create_animation
import sys

parser = argparse.ArgumentParser(description='Generate videos from dataset')
parser.add_argument('inp',help='input directory')
parser.add_argument('out',help='output directory')

args = parser.parse_args()

if not (os.path.isdir(args.inp)):
    print("Invalid input directory")
    sys.exit()

if not (os.path.isdir(args.out)):
    try:
        os.mkdir(args.out)
    except:
        print("Error creating output directory")
        sys.exit()

for file in os.listdir(args.inp):

    # load the data
    pkl = os.path.join(args.inp, file)
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

    create_animation(buyerJoints, leftSellerJoints, rightSellerJoints, os.path.join(args.out, file.split('.')[0]))
