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
flags.DEFINE_string('input', './preprocessed', 'Input Directory')
flags.DEFINE_string('output', './normalized', 'Output Directory')


def create_animation(buyerJoints, leftSellerJoints, rightSellerJoints, fileName):
    """
    Generates a video file of the subjects' motion
    :param buyerJoints: 'joints19' of buyer of shape (keypoints, seqLength)
    :param leftSellerJoints: 'joints19' of left seller of shape (keypoints, seqLength)
    :param rightSellerJoints: 'joints19' of right seller of shape (keypoints, seqLength)
    :param fileName: fileName of the output animation
    :return: None
    """

    # create the connection list for skeleton
    humanSkeleton = [
        [0,1],
        [0,3],
        [3,4],
        [4,5],
        [0,2],
        [2,6],
        [6,7],
        [7,8],
        [2,12],
        [12,13],
        [13,14],
        [0,9],
        [9,10],
        [10,11]
    ]

    # add the joints and the marker colors
    # setup the frame layout
    fig = plt.figure()
    plt.axis('off')
    ax = p3.Axes3D(fig)
    ax.view_init(elev=-60, azim=-90)
    ax._axis3don = False
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # set up the writer
    # Set up formatting for the movie files
    Writer = anim.writers['ffmpeg']
    writer = Writer(fps=24, metadata=dict(artist='Kurel002'), bitrate=1800)

    jointList = []
    leftSellerPoints = []
    for i in range(0, len(leftSellerJoints), 3):
        x = leftSellerJoints[i][0]
        y = leftSellerJoints[i + 1][0]
        z = leftSellerJoints[i + 2][0]
        jointList.append((x, y, z))
        graph, = ax.plot([x], [y], [z], linestyle="", c='r', marker="o")
        leftSellerPoints.append(graph)

    leftSellerBones = []
    for start, end in humanSkeleton:
        xs = [jointList[start][0], jointList[end][0]]
        ys = [jointList[start][1], jointList[end][1]]
        zs = [jointList[start][2], jointList[end][2]]
        graph, = ax.plot(xs, ys, zs, c='r')
        leftSellerBones.append(graph)

    jointList = []
    rightSellerPoints = []
    for i in range(0, len(rightSellerJoints), 3):
        x = rightSellerJoints[i][0]
        y = rightSellerJoints[i + 1][0]
        z = rightSellerJoints[i + 2][0]
        jointList.append((x, y, z))
        graph, = ax.plot([x], [y], [z], linestyle="", c='b', marker="o")
        rightSellerPoints.append(graph)

    rightSellerBones = []
    for start, end in humanSkeleton:
        xs = [jointList[start][0], jointList[end][0]]
        ys = [jointList[start][1], jointList[end][1]]
        zs = [jointList[start][2], jointList[end][2]]
        graph, = ax.plot(xs, ys, zs, c='b')
        rightSellerBones.append(graph)

    jointList = []
    buyerPoints = []
    for i in range(0, len(buyerJoints), 3):
        x = buyerJoints[i][0]
        y = buyerJoints[i + 1][0]
        z = buyerJoints[i + 2][0]
        jointList.append((x, y, z))
        graph, = ax.plot([x], [y], [z], linestyle="", c='k', marker="o")
        buyerPoints.append(graph)

    buyerBones = []
    for start, end in humanSkeleton:
        xs = [jointList[start][0], jointList[end][0]]
        ys = [jointList[start][1], jointList[end][1]]
        zs = [jointList[start][2], jointList[end][2]]
        graph, = ax.plot(xs, ys, zs, c='k')
        buyerBones.append(graph)

    def update_plot(frame, joints, points, bones, humanSkeleton):

        leftSellerPoints = points[0]
        rightSellerPoints = points[1]
        buyerPoints = points[2]

        leftSellerJoints = joints[0]
        rightSellerJoints = joints[1]
        buyerJoints = joints[2]

        leftSellerBones = bones[0]
        rightSellerBones = bones[1]
        buyerBones = bones[2]

        jointList = []
        for i in range(len(leftSellerPoints)):
            graph = leftSellerPoints[i]
            x = leftSellerJoints[3 * i][frame]
            y = leftSellerJoints[3 * i + 1][frame]
            z = leftSellerJoints[3 * i + 2][frame]
            jointList.append((x, y, z))
            graph.set_data([x], [y])
            graph.set_3d_properties([z])

        for i in range(len(leftSellerBones)):
            start, end = humanSkeleton[i]
            graph = leftSellerBones[i]
            xs = [jointList[start][0], jointList[end][0]]
            ys = [jointList[start][1], jointList[end][1]]
            zs = [jointList[start][2], jointList[end][2]]
            graph.set_data(xs, ys)
            graph.set_3d_properties(zs)

        jointList = []
        for i in range(len(rightSellerPoints)):
            graph = rightSellerPoints[i]
            x = rightSellerJoints[3 * i][frame]
            y = rightSellerJoints[3 * i + 1][frame]
            z = rightSellerJoints[3 * i + 2][frame]
            jointList.append((x, y, z))
            graph.set_data([x], [y])
            graph.set_3d_properties([z])

        for i in range(len(rightSellerBones)):
            graph = rightSellerBones[i]
            start, end = humanSkeleton[i]
            xs = [jointList[start][0], jointList[end][0]]
            ys = [jointList[start][1], jointList[end][1]]
            zs = [jointList[start][2], jointList[end][2]]
            graph.set_data(xs, ys)
            graph.set_3d_properties(zs)

        jointList = []
        for i in range(len(buyerPoints)):
            graph = buyerPoints[i]
            x = buyerJoints[3 * i][frame]
            y = buyerJoints[3 * i + 1][frame]
            z = buyerJoints[3 * i + 2][frame]
            jointList.append((x, y, z))
            graph.set_data([x], [y])
            graph.set_3d_properties([z])

        for i in range(len(buyerBones)):
            graph = buyerBones[i]
            start, end = humanSkeleton[i]
            xs = [jointList[start][0], jointList[end][0]]
            ys = [jointList[start][1], jointList[end][1]]
            zs = [jointList[start][2], jointList[end][2]]
            graph.set_data(xs, ys)
            graph.set_3d_properties(zs)

    points = [leftSellerPoints, rightSellerPoints, buyerPoints]
    joints = [leftSellerJoints, rightSellerJoints, buyerJoints]
    bones = [leftSellerBones, rightSellerBones, buyerBones]
    frames = len(leftSellerJoints[0])

    ani = anim.FuncAnimation(fig, update_plot, frames, fargs=(joints, points, bones, humanSkeleton), interval=50, repeat=True)

    # save only if filename is passed
    if fileName:
        print(fileName+'.mp4')
        ani.save(fileName + '.mp4', writer=writer)
    else:
        plt.show()

    # close plot
    plt.close()

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
                
        create_animation(buyerJoints, leftSellerJoints, rightSellerJoints, file)

        # Remove .pkl extension from name
        file = Path(str(file))
        file = str(file.with_suffix(''))
        
        # Write output file
        outFile = open(('%s/%s.pkl' % (FLAGS.output, file)), 'wb')
        print("Writing data from %s to file %s_norm.pkl" % (file, file))
        pickle.dump(new_pickle, outFile)
        outFile.close()
        break

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