import pickle
import os
import argparse
import sys
import math
import numpy as np

parser = argparse.ArgumentParser(description='Preprocess videos from dataset, removing contaminated segments')
parser.add_argument('inp',help='input pkl file')
parser.add_argument('out',help='output directory')

args = parser.parse_args()

if not (os.path.isfile(args.inp)):
    print("Invalid input pkl file")
    sys.exit()

if not (os.path.isdir(args.out)):
    try:
        os.mkdir(args.out)
    except:
        print("Error creating output directory")
        sys.exit()

        
# load the data
pkl = open(args.inp, 'rb')
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

# Each subject has 'joints19' attribute with x-y-z coords for each joint
# Each coordinate consists of a list of position values corresponding to a frame
# Collect all of the joint positions for each frame, for later examination
frame_joints = []
for f in range(0, len(leftSellerJoints[0])):    
    jointList = []
    for i in range(0, len(leftSellerJoints), 3):
        x = leftSellerJoints[i][f]
        y = leftSellerJoints[i + 1][f]
        z = leftSellerJoints[i + 2][f]
        jointList.append((x, y, z))

#    for i in range(0, len(rightSellerJoints), 3):
#        x = leftSellerJoints[i][f]
#        y = leftSellerJoints[i + 1][f]
#        z = leftSellerJoints[i + 2][f]
#        jointList.append((x, y, z))
#
#    for i in range(0, len(buyerJoints), 3):
#        x = leftSellerJoints[i][f]
#        y = leftSellerJoints[i + 1][f]
#        z = leftSellerJoints[i + 2][f]
#        jointList.append((x, y, z))

    frame_joints.append(jointList)

print("Frame len: ", len(frame_joints), ", single frame: ", len(frame_joints[0]), ", single joint: ", len(frame_joints[0][0]))

# Collect the lengths of each bone, giving a baseline expectation for each length. 
base = []
for start, end in humanSkeleton:
        xs = frame_joints[0][start][0] - frame_joints[0][end][0]
        ys = frame_joints[0][start][1] - frame_joints[0][end][1]
        zs = frame_joints[0][start][2] - frame_joints[0][end][2]
        base.append(math.sqrt(xs**2 + ys**2 + zs**2))

# Within each frame, check if the bone lengths fall within expectations
frame_bones = []
problem_frame_joints = []
for f in range(1, len(frame_joints)):
    bone_lengths = []
    faulty = False

    # Check each bone within the frame
    for i in range(0, len(humanSkeleton)):
        start, end = humanSkeleton[i]
        xs = frame_joints[f][start][0] - frame_joints[f][end][0]
        ys = frame_joints[f][start][1] - frame_joints[f][end][1]
        zs = frame_joints[f][start][2] - frame_joints[f][end][2]
        length = math.sqrt(xs**2 + ys**2 + zs**2)
        bone_lengths.append(length)
        if (abs(length - base[i]) > 2):
            faulty = True
    if (faulty):
        problem_frame_joints.append(f)
    frame_bones.append(bone_lengths)

maxes = []
frame_bones = np.array(frame_bones)
print(frame_bones.shape)
print("%d faulty frames" % (len(problem_frame_joints)))
for i in range(0, len(frame_bones[0])):
    print(len(frame_bones[i]))
#    print(np.mean(frame_bones, axis=i))
print(base)
# for i in range(0, 100):
#    print(frame_bones[i][0])
for i in frame_bones:
    print(i[0])
