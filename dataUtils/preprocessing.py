import copy
import math
import os
import pickle
import sys
import numpy as np
from absl import flags
from absl import app

# Set up flags
FLAGS = flags.FLAGS
flags.DEFINE_string("input", "../input", "Input pkl directory")
flags.DEFINE_string("output", "../preprocessed", "Output Directory")


def joint_positions(leftSellerJoints, rightSellerJoints, buyerJoints):
    """
    Find the x-y-z coordinates for each person"s joints within each frame from the original frame data.
    :param leftSellerJoints: Joint information for the left seller
    :param rightSellerJoints: Joint information for the right seller
    :param buyerJoints: Joint information for the buyer
    :return: Lists of x-y-z coordinate triples for the left seller, right seller, and buyer, respectively.
    """
    # Each subject has "joints19" attribute with x-y-z coords for each joint
    # Each coordinate consists of a list of position values corresponding to a frame
    # Collect all of the joint positions for each frame, for later examination
    leftSellerFrameJoints = []
    rightSellerFrameJoints = []
    buyerFrameJoints = []
    for f in range(0, len(leftSellerJoints[0])):
        # Collect joint positions for left seller
        jointList = []
        for i in range(0, len(leftSellerJoints), 3):
            x = leftSellerJoints[i][f]
            y = leftSellerJoints[i + 1][f]
            z = leftSellerJoints[i + 2][f]
            jointList.append((x, y, z))
        leftSellerFrameJoints.append(jointList)

        # Collect joint positions for right seller
        jointList = []
        for i in range(0, len(rightSellerJoints), 3):
            x = rightSellerJoints[i][f]
            y = rightSellerJoints[i + 1][f]
            z = rightSellerJoints[i + 2][f]
            jointList.append((x, y, z))
        rightSellerFrameJoints.append(jointList)

        # Collect joint positions for buyer
        jointList = []
        for i in range(0, len(buyerJoints), 3):
            x = buyerJoints[i][f]
            y = buyerJoints[i + 1][f]
            z = buyerJoints[i + 2][f]
            jointList.append((x, y, z))
        buyerFrameJoints.append(jointList)

    return leftSellerFrameJoints, rightSellerFrameJoints, buyerFrameJoints


def bone_lengths(lSJoints, rSJoints, bJoints, humanSkeleton):
    """
    Calculate and collect the float length of each bone into a single list.
    :param lSJoints: Raw joint data for the left seller
    :param rSJoints: Raw joint data for the right seller
    :param bJoints: Raw joint data for the buyer
    :param humanSkeleton: List of joint pairs, representing the joint connections (bones) in our person models.
    :return: A list of length triples, representing the lengths of each person"s bones. Example format is:
    Frame k: [leftBone1, rightBone1, buyerBone1, leftBone2, rightBone2, buyerBone2, leftBone3, ...]
    """

    # Determine the positions of each person"s joints
    leftSellerFrameJoints, rightSellerFrameJoints, buyerFrameJoints = joint_positions(lSJoints, rSJoints, bJoints)

    # Calculate the length of each person"s bones
    frameBones = []
    for f in range(0, len(leftSellerFrameJoints)):
        # Record length of each person"s bones within each frame
        # Keep the lengths in a flatter array to avoid timing issues later.
        boneLengths = []
        for i in range(0, len(humanSkeleton)):
            start, end = humanSkeleton[i]

            # Calculate lengths for left seller
            xs = leftSellerFrameJoints[f][start][0] - leftSellerFrameJoints[f][end][0]
            ys = leftSellerFrameJoints[f][start][1] - leftSellerFrameJoints[f][end][1]
            zs = leftSellerFrameJoints[f][start][2] - leftSellerFrameJoints[f][end][2]
            boneLengths.append(math.sqrt(xs ** 2 + ys ** 2 + zs ** 2))

            # Calculate lengths for right seller
            xs = rightSellerFrameJoints[f][start][0] - rightSellerFrameJoints[f][end][0]
            ys = rightSellerFrameJoints[f][start][1] - rightSellerFrameJoints[f][end][1]
            zs = rightSellerFrameJoints[f][start][2] - rightSellerFrameJoints[f][end][2]
            boneLengths.append(math.sqrt(xs ** 2 + ys ** 2 + zs ** 2))

            # Calculate lengths for buyer
            xs = buyerFrameJoints[f][start][0] - buyerFrameJoints[f][end][0]
            ys = buyerFrameJoints[f][start][1] - buyerFrameJoints[f][end][1]
            zs = buyerFrameJoints[f][start][2] - buyerFrameJoints[f][end][2]
            boneLengths.append(math.sqrt(xs ** 2 + ys ** 2 + zs ** 2))

        frameBones.append(boneLengths)

    return frameBones


def get_segments(bones):
    """
    Determines which segments of a video"s data are of acceptable quality and length.
    :param bones: Input bone length data for the video"s frames.
    :return: List of pairs, describing the start and end frames of each acceptable segment.
    """
    # Within each frame, check if the bone lengths fall inside our tolerance threshold
    # If not, add the frame to a list of bad frames for later use
    problemFrames = []
    frameBones = np.array(bones)  # Convert to numpy array find distribution stats easier
    means = np.mean(frameBones, axis=0)
    tolerance = np.std(frameBones, axis=0) * 3
    for f in range(0, len(frameBones)):
        for i in range(0, len(frameBones[f])):
            if frameBones[f][i] < means[i] - tolerance[i] or frameBones[f][i] > means[i] + tolerance[i]:
                problemFrames.append(f)
                break
    print("%d faulty frames" % (len(problemFrames)))
    # print("Bone mean of the lengths: ", np.mean(frameBones, axis=0))
    # print("Bone standard deviation of the lengths: ", np.std(frameBones, axis=0))

    # Define partitions for writing out files as the start and end frames numbers of whole segments.
    segments = []
    for target in range(0, len(problemFrames) - 1):
        # If there"s a sequence of frames longer than 100 frames, add to the accepted segments.
        if problemFrames[target + 1] - problemFrames[target] > 100:
            segments.append((problemFrames[target], problemFrames[target + 1]))

    return segments


def preprocess_frames(inp, output):
    """
    Given some "raw" video data as a pickle file, produces some number of files containing the frame data for
    exactly one acceptable segment of video. Number of files produced depends upon the quality and quantity of the
    original"s video data.
    :param inp: "Raw" pickle file"s path
    :param output: Path of the directory where the new pickle file(s) will be stored
    :return: None
    """

    # load the data
    pkl = open(inp, "rb")
    group = pickle.load(pkl)

    # load buyer, seller Ids
    leftSellerId = group["leftSellerId"]
    rightSellerId = group["rightSellerId"]
    buyerId = group["buyerId"]

    # load skeletons
    buyerJoints = []
    leftSellerJoints = []
    rightSellerJoints = []

    # load the skeletons
    for subject in group["subjects"]:
        if subject["humanId"] == leftSellerId:
            leftSellerJoints = subject["joints19"]
        if subject["humanId"] == rightSellerId:
            rightSellerJoints = subject["joints19"]
        if subject["humanId"] == buyerId:
            buyerJoints = subject["joints19"]

    # Define the relevant connections ("bones") between the joints to visualize
    humanSkeleton = [
        [0, 1],
        [0, 3],
        [3, 4],
        [4, 5],
        [0, 2],
        [2, 6],
        [6, 7],
        [7, 8],
        [2, 12],
        [12, 13],
        [13, 14],
        [0, 9],
        [9, 10],
        [10, 11]
    ]

    # Calculate each person"s bone lengths within each frame
    frameBones = bone_lengths(leftSellerJoints, rightSellerJoints, buyerJoints, humanSkeleton)

    # Use the bone length data to filter out outliers and excessively noisy data
    segments = get_segments(frameBones)

    # Create a new copy of the data, to be reused for each data segment
    new_pickle = copy.deepcopy(group)

    # Write new pickle files, each one containing data for exactly one good segment of video
    fileCount = 0
    for seg in segments:

        # Record data for each subject in the video
        for i in range(0, len(group["subjects"])):
            # Reset sizes of the 2d numpy arrays in the new pickle dictionary
            new_subject = new_pickle["subjects"][i]
            new_subject["bodyNormal"] = np.zeros((3, seg[1] - seg[0]))
            new_subject["faceNormal"] = np.zeros((3, seg[1] - seg[0]))
            new_subject["scores"] = np.zeros((19, seg[1] - seg[0]))
            new_subject["joints19"] = np.zeros((57, seg[1] - seg[0]))
            old_subject = group["subjects"][i]

            # Use trimmed segments from original data for all frame-dependent information
            for j in range(0, 3):
                new_subject["bodyNormal"][j] = old_subject["bodyNormal"][j][seg[0]:seg[1]]
                new_subject["faceNormal"][j] = old_subject["faceNormal"][j][seg[0]:seg[1]]
            for j in range(0, 19):
                new_subject["scores"][j] = old_subject["scores"][j][seg[0]:seg[1]]
            for j in range(0, 57):
                new_subject["joints19"][j] = old_subject["joints19"][j][seg[0]:seg[1]]

        # Write the trimmed data to the designated files
        base = os.path.basename(inp)
        outFile = open(("%s/%s_part%d.pkl" % (output, base[:-4], fileCount)), "wb")
        print("Writing data from frames %d:%d to file %s_part%d.pkl" % (seg[0], seg[1], base[:-4], fileCount))
        pickle.dump(new_pickle, outFile)
        outFile.close()
        fileCount += 1


def main(argv):
    """
    Generates videos from raw data set.
    :param argv: Input arguments
    :return: None
    """
    # Check if the input directory exists
    if not (os.path.isdir(FLAGS.input)):
        print("Invalid input pkl directory")
        sys.exit()

    # Create the output directory if none exist
    if not (os.path.isdir(FLAGS.output)):
        try:
            os.mkdir(FLAGS.output)
        except:
            print("Error creating output directory")
            sys.exit()

    # For each pickle file within the input directory, preprocess its data
    for file in os.listdir(FLAGS.input):
        target = "%s/%s" % (FLAGS.input, file)
        print("Reading data from: ", target)
        preprocess_frames(target, FLAGS.output)


if __name__ == "__main__":
    app.run(main)
