import copy
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
import os
import pickle
import sys
import tensorflow as tf
from absl import app
from absl import flags
from pathlib import Path

# set up flags
FLAGS = flags.FLAGS

# TODO change input dir to preprocessed 
flags.DEFINE_string('input', '../output', 'Input Directory')
flags.DEFINE_string('output', '../normalized', 'Output Directory')


def get_min_max(leftSellerJoints, rightSellerJoints, buyerJoints):
    '''
    Returns min and max of X Y Z coordinates 
    works on leftSeller, rightSeller and buyerJoints
    :param all the joints for each pkl file
    :return min values x y z, max values x y z
    '''
    min_left_x = np.amin(leftSellerJoints[::3][:])
    min_left_y = np.amin(leftSellerJoints[1::3][:])
    min_left_z = np.amin(leftSellerJoints[2::3][:])
    
    min_right_x = np.amin(rightSellerJoints[::3][:])
    min_right_y = np.amin(rightSellerJoints[1::3][:])
    min_right_z = np.amin(rightSellerJoints[2::3][:])
    
    min_buyer_x = np.amin(buyerJoints[::3][:])
    min_buyer_y = np.amin(buyerJoints[1::3][:])
    min_buyer_z = np.amin(buyerJoints[2::3][:])
    
    max_left_x = np.amax(leftSellerJoints[::3][:])
    max_left_y = np.amax(leftSellerJoints[1::3][:])
    max_left_z = np.amax(leftSellerJoints[2::3][:])
    
    max_right_x = np.amax(rightSellerJoints[::3][:])
    max_right_y = np.amax(rightSellerJoints[1::3][:])
    max_right_z = np.amax(rightSellerJoints[2::3][:])
    
    max_buyer_x = np.amax(buyerJoints[::3][:])
    max_buyer_y = np.amax(buyerJoints[1::3][:])
    max_buyer_z = np.amax(buyerJoints[2::3][:])
    
    min_x = min(min_left_x, min_right_x, min_buyer_x)
    max_x = max(max_left_x, max_right_x, max_buyer_x)
    
    min_y = min(min_left_y, min_right_y, min_buyer_y)
    max_y = max(max_left_y, max_right_y, max_buyer_y)
    
    min_z = min(min_left_z, min_right_z, min_buyer_z)
    max_z = max(max_left_z, max_right_z, max_buyer_z)
    
    return min_x, min_y, min_z, max_x, max_y, max_z


def get_min_max_for_pkl_file(folder_name, file_name):
    '''
    Returns min and max of pkl files being worked on
    :param name of the work folder, name of the file worked on
    :return min values x y z, global max values x y z
    '''
    pkl = os.path.join(folder_name, file_name)
    pkl = open(pkl, 'rb')
    group = pickle.load(pkl)

    # load buyer, seller Ids
    leftSellerId = group['leftSellerId']
    rightSellerId = group['rightSellerId']
    buyerId = group['buyerId']

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
            
    min_x, min_y, min_z, max_x, max_y, max_z = get_min_max(leftSellerJoints, rightSellerJoints, buyerJoints)
    return min_x, min_y, min_z, max_x, max_y, max_z


def get_global_min_max():
    '''
    Returns the global min and global max for all pkl files in a folder
    :param none
    :return the global min x y z, the global min x y z
    '''
    g_min_x, g_min_y, g_min_z = float("inf"), float("inf"), float("inf")
    g_max_x, g_max_y, g_max_z = -float("inf"), -float("inf"), -float("inf")
    
    # Iterate through all the pkl files in the work folder
    for file in os.listdir(FLAGS.input):
        min_x, min_y, min_z, max_x, max_y, max_z = get_min_max_for_pkl_file(FLAGS.input, file)
        g_min_x = min(min_x, g_min_x)
        g_min_y = min(min_y, g_min_y)
        g_min_z = min(min_z, g_min_z)
        g_max_x = max(max_x, g_max_x)
        g_max_y = max(max_y, g_max_y)
        g_max_z = max(max_z, g_max_z)
    return g_min_x, g_min_y, g_min_z, g_max_x, g_max_y, g_max_z


def normalize_pkl_files():
    '''
    Normalizes all the pickle files in the input folder
    - gets the global min and max for x y z
    - normalizes the skeletons based on global min max
    - writes it back to a new pickle file
    :param none
    :return none, generates output files
    '''
    min_x, min_y, min_z, max_x, max_y, max_z = get_global_min_max()
    for file in os.listdir(FLAGS.input):
        pkl = os.path.join(FLAGS.input, file)
        pkl = open(pkl, 'rb')
        group = pickle.load(pkl)
        
        # To store the normalized data
        new_pickle = copy.deepcopy(group)

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
                
        # Min Max normalization
        
        leftSellerJoints[::3][:] -= min_x
        leftSellerJoints[::3][:] /= (max_x - min_x)
        
        leftSellerJoints[1::3][:] -= min_y
        leftSellerJoints[1::3][:] /= (max_y - min_y)

        leftSellerJoints[2::3][:] -= min_z
        leftSellerJoints[2::3][:] /= (max_z - min_z)
    

        rightSellerJoints[::3][:] -= min_x
        rightSellerJoints[::3][:] /= (max_x - min_x)
        
        rightSellerJoints[1::3][:] -= min_y
        rightSellerJoints[1::3][:] /= (max_y - min_y)
        
        rightSellerJoints[2::3][:] -= min_z
        rightSellerJoints[2::3][:] /= (max_z - min_z)

        buyerJoints[::3][:] -= min_x
        buyerJoints[::3][:] /= (max_x - min_x)
        
        buyerJoints[1::3][:] -= min_y
        buyerJoints[1::3][:] /= (max_y - min_y)
        
        buyerJoints[2::3][:] -= min_z
        buyerJoints[2::3][:] /= (max_z - min_z)
        
        # update the skeletons with normalized data
        for subject in new_pickle['subjects']:
            if (subject['humanId'] == leftSellerId):
                subject['joints19'] = leftSellerJoints
            if (subject['humanId'] == rightSellerId):
                subject['joints19'] = rightSellerJoints
            if (subject['humanId'] == buyerId):
                subject['joints19'] = buyerJoints
                
        # Remove .pkl extension from name
        file = Path(str(file))
        file = str(file.with_suffix(''))
        
        # Write output file
        outFile = open(('%s/%s.pkl' % (FLAGS.output, file)), 'wb')
        print("Writing data from %s to file %s_norm.pkl" % (file, file))
        pickle.dump(new_pickle, outFile)
        outFile.close()


def main(argv):
    """
    Performs min max normalization on pickle files 
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
            
    normalize_pkl_files()


if __name__ == '__main__':
    app.run(main)