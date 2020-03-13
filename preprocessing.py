import pickle
import os
import argparse
import sys
import math
import numpy as np
import copy

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
leftSellerFrameJoints = []
rightSellerFrameJoints = []
buyerFrameJoints = []
for f in range(0, len(leftSellerJoints[0])):    
    jointList = []
    for i in range(0, len(leftSellerJoints), 3):
        x = leftSellerJoints[i][f]
        y = leftSellerJoints[i + 1][f]
        z = leftSellerJoints[i + 2][f]
        jointList.append((x, y, z))
    leftSellerFrameJoints.append(jointList)

    jointList = []
    for i in range(0, len(rightSellerJoints), 3):
        x = rightSellerJoints[i][f]
        y = rightSellerJoints[i + 1][f]
        z = rightSellerJoints[i + 2][f]
        jointList.append((x, y, z))
    rightSellerFrameJoints.append(jointList)

    jointList = []
    for i in range(0, len(buyerJoints), 3):
        x = buyerJoints[i][f]
        y = buyerJoints[i + 1][f]
        z = buyerJoints[i + 2][f]
        jointList.append((x, y, z))
    buyerFrameJoints.append(jointList)

# Collect the lengths of each bone, giving a baseline expectation for each length. 
base = []
for start, end in humanSkeleton:
        xs = leftSellerFrameJoints[0][start][0] - leftSellerFrameJoints[0][end][0]
        ys = leftSellerFrameJoints[0][start][1] - leftSellerFrameJoints[0][end][1]
        zs = leftSellerFrameJoints[0][start][2] - leftSellerFrameJoints[0][end][2]
        base.append(math.sqrt(xs**2 + ys**2 + zs**2))

# Collect the length of each bone into a single list of lists
# Format will be: leftBone1, rightBone1, buyerBone1, leftBone2, ...
frameBones = []
for f in range(0, len(leftSellerFrameJoints)):
    boneLengths = []
    
    # Record length of each person's bones within the frame
    # Keep the lengths in a flatter array to avoid timing issues later.
    for i in range(0, len(humanSkeleton)):
        start, end = humanSkeleton[i]
        xs = leftSellerFrameJoints[f][start][0] - leftSellerFrameJoints[f][end][0]
        ys = leftSellerFrameJoints[f][start][1] - leftSellerFrameJoints[f][end][1]
        zs = leftSellerFrameJoints[f][start][2] - leftSellerFrameJoints[f][end][2]
        boneLengths.append(math.sqrt(xs**2 + ys**2 + zs**2))
        xs = rightSellerFrameJoints[f][start][0] - rightSellerFrameJoints[f][end][0]
        ys = rightSellerFrameJoints[f][start][1] - rightSellerFrameJoints[f][end][1]
        zs = rightSellerFrameJoints[f][start][2] - rightSellerFrameJoints[f][end][2]
        boneLengths.append(math.sqrt(xs**2 + ys**2 + zs**2))
        xs = buyerFrameJoints[f][start][0] - buyerFrameJoints[f][end][0]
        ys = buyerFrameJoints[f][start][1] - buyerFrameJoints[f][end][1]
        zs = buyerFrameJoints[f][start][2] - buyerFrameJoints[f][end][2]
        boneLengths.append(math.sqrt(xs**2 + ys**2 + zs**2))
    frameBones.append(boneLengths)

# Within each frame, check if the bone lengths fall within expectations
# If not, add the frame to a list of bad frames for later use
problemFrames = []
frameBones = np.array(frameBones)  # Convert to find distribution stats easier
means = np.mean(frameBones, axis=0)
deviations = np.std(frameBones, axis=0)
for f in range(0, len(frameBones)):
    faulty = False
    for i in range(0, len(frameBones[f])):
        if (frameBones[f][i] < means[i] - 3 * deviations[i] or frameBones[f][i] > means[i] + 3 * deviations[i]):
            problemFrames.append(f)
            faulty = True
            break
print("%d faulty frames" % (len(problemFrames)))
        # print("Bone mean lengths: ", np.mean(frameBones, axis=0))
        # print("Bone std devs lengths: ", np.std(frameBones, axis=0))
        
# Create partitions for writing out files, containing long, whole segments of good frames.
# Specifically records which frames are targeted, then the x-y-z coordinates of each joint
# for each person for each frame, in chronological order. 
segments = []
for target in range(0, len(problemFrames) - 1):
    # If there's a sequence of frames longer than 100 frames, add to the accepted segments.
    if (problemFrames[target + 1] - problemFrames[target] > 100):
        segments.append((problemFrames[target], problemFrames[target+1]))

# Create a new copy of the data, to be reused for each data segment
new_pickle = copy.deepcopy(group)
fileCount = 0
for seg in segments:
    for i in range(0, len(group['subjects'])):
        # Reset sizes of the 2d numpy arrays in the new pickle dictionary
        new_pickle['subjects'][i]['bodyNormal'] = np.zeros((3, seg[1] - seg[0]))
        new_pickle['subjects'][i]['faceNormal'] = np.zeros((3, seg[1] - seg[0]))
        new_pickle['subjects'][i]['scores'] = np.zeros((19, seg[1] - seg[0]))
        new_pickle['subjects'][i]['joints19'] = np.zeros((57, seg[1] - seg[0]))

        # Use trimmed segments from original data
        for j in range(0, 3):
            new_pickle['subjects'][i]['bodyNormal'][j] = group['subjects'][i]['bodyNormal'][j][seg[0]:seg[1]]
            new_pickle['subjects'][i]['faceNormal'][j] = group['subjects'][i]['faceNormal'][j][seg[0]:seg[1]]
        for j in range(0, 19):
            new_pickle['subjects'][i]['scores'][j] = group['subjects'][i]['scores'][j][seg[0]:seg[1]]
        for j in range(0, 57):
            new_pickle['subjects'][i]['joints19'][j] = group['subjects'][i]['joints19'][j][seg[0]:seg[1]]

    # Write the trimmed data to the designated files
    base = os.path.basename(args.inp)
    outFile = open(('%s/%s_part%d.txt' % (args.out, base[:-4], fileCount)), 'wb')
    print("Writing data from frames %d:%d to file %s_part%d.txt" % (seg[0], seg[1], base[:-4], fileCount))
    pickle.dump(new_pickle, outFile)
    outFile.close()
    fileCount += 1
