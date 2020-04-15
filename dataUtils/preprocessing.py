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
flags.DEFINE_integer("subsample", 0, "How many frames to skip")
flags.DEFINE_integer("supersample", 0, "How many frames to linearly interpolate betewen recorded data")
flags.DEFINE_integer("min_length", 50, "How many frames are required to consider a video clip long enough to use")


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
    for frame in range(0, len(leftSellerFrameJoints)):
        # Record length of each person"s bones within each frame
        # Keep the lengths in a flatter array to avoid timing issues later.
        boneLengths = []
        for bone in range(0, len(humanSkeleton)):
            start, end = humanSkeleton[bone]

            # Calculate lengths for left seller
            xs = leftSellerFrameJoints[frame][start][0] - leftSellerFrameJoints[frame][end][0]
            ys = leftSellerFrameJoints[frame][start][1] - leftSellerFrameJoints[frame][end][1]
            zs = leftSellerFrameJoints[frame][start][2] - leftSellerFrameJoints[frame][end][2]
            boneLengths.append(math.sqrt(xs ** 2 + ys ** 2 + zs ** 2))

            # Calculate lengths for right seller
            xs = rightSellerFrameJoints[frame][start][0] - rightSellerFrameJoints[frame][end][0]
            ys = rightSellerFrameJoints[frame][start][1] - rightSellerFrameJoints[frame][end][1]
            zs = rightSellerFrameJoints[frame][start][2] - rightSellerFrameJoints[frame][end][2]
            boneLengths.append(math.sqrt(xs ** 2 + ys ** 2 + zs ** 2))

            # Calculate lengths for buyer
            xs = buyerFrameJoints[frame][start][0] - buyerFrameJoints[frame][end][0]
            ys = buyerFrameJoints[frame][start][1] - buyerFrameJoints[frame][end][1]
            zs = buyerFrameJoints[frame][start][2] - buyerFrameJoints[frame][end][2]
            boneLengths.append(math.sqrt(xs ** 2 + ys ** 2 + zs ** 2))

        frameBones.append(boneLengths)

    return frameBones


def velocities(leftSellerJoints, rightSellerJoints, buyerJoints):
    """
    Determine changes in position for each subject's joints in the video
    :param leftSellerJoints: The left seller's "joints19" data from the pickle files
    :param rightSellerJoints: The right seller's "joints19" data from the pickle files
    :param buyerJoints: The buyer's "joints19" data from the pickle files
    :return: The change in position for each joint from frame i to i + 1, as a
    triple of x-y-z deltas in the form of:
    Frame f: [lS(joint1(dx, dy, dz), ..., joint19(dx, dy, dz)),
    rS(joint1(dx, dy, dz), ..., joint19(dx, dy, dz),
    b(joint1(dx, dy, dz), ..., joint19(dx, dy, dz))]
    """
    # Partition position data into left, right, and buyer framewise joints
    # Use np array format for easy array arithmetic
    leftSellerFrameJoints, rightSellerFrameJoints, buyerFrameJoints = np.array(joint_positions(leftSellerJoints,
                                                                                               rightSellerJoints,
                                                                                               buyerJoints))

    # Calculate the changes in each subject's joint positions throughout the video, then append to the master array
    total = []
    for frame in range(1, leftSellerJoints.shape[1]):
        d_leftSeller = leftSellerFrameJoints[frame] - leftSellerFrameJoints[frame - 1]
        d_rightSeller = rightSellerFrameJoints[frame] - rightSellerFrameJoints[frame - 1]
        d_buyer = buyerFrameJoints[frame] - buyerFrameJoints[frame - 1]
        total.append((d_leftSeller, d_rightSeller, d_buyer))

    total = np.array(total)
    return total


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

    # Current method uses mean and standard deviations for the error tolerances
    means = np.mean(frameBones, axis=0)
    tolerance = np.std(frameBones, axis=0) * 5

    # Alternative method: Use percentiles as thresholds
    # upper_percent = np.percentile(frameBones, 99, axis=0)
    # lower_percent = np.percentile(frameBones, 1, axis=0)

    for frame in range(0, len(frameBones)):
        for joint in range(0, len(frameBones[frame])):
            lower = means[joint] - tolerance[joint]
            upper = means[joint] + tolerance[joint]
            # upper = upper_percent[joint]
            # lower = lower_percent[joint]
            if np.any(lower > frameBones[frame][joint]) or np.any(upper < frameBones[frame][joint]):
                problemFrames.append(frame)
                break

    # Define partitions for writing out files as the start and end frames numbers of whole segments.
    segments = []
    for target in range(len(problemFrames) - 1):
        # If a sequence is more than the desired length, add the trimmed version to those accepted.
        if problemFrames[target + 1] - problemFrames[target] > FLAGS.min_length:
            segments.append((problemFrames[target]+1, problemFrames[target + 1]))

    # If no problem frames found, write all of the video's data out
    if len(problemFrames) == 0:
        segments.append((0, len(frameBones) - 1))

    return segments


def super_sampling(inp, k):
    """
    Returns a linearly interpolated supersample of the original sequence
    :param inp: Input sequence of positions
    :param k: Number of interpolated frames, should be an integer >= 1
    :return: The newly generated sequence of (k + 1) * n frames
    """
    out = []
    for frame in range(1, len(inp)):
        diff = inp[frame] - inp[frame - 1]   # Find the average change in position between frames
        for step in range(int(k)):
            out.append(step / k * diff + inp[frame - 1])
    return np.array(out)


def sub_sampling(inp, k):
    """
    Returns a subsample which skips every kth frame of the original sequence
    :param inp: Input sequence of posiitons as a numpy array
    :param k: Number of frames to be skipped at each step
    :return: The newly generated sequence of n / k frames
    """
    out = []
    for frame in range(0, inp.shape[0], k):
        out.append(inp[frame])
    return np.array(out)


def preprocess_frames(inp, output, subsample, supersample):
    """
    Given some "raw" video data as a pickle file, produces some number of files containing the frame data for
    exactly one acceptable segment of video. Number of files produced depends upon the quality and quantity of the
    original"s video data.
    :param inp: "Raw" pickle file"s path
    :param output: Path of the directory where the new pickle file(s) will be stored
    :param subsample: How many frames to skip within an input sequence
    :param supersample: How many frames to interpolate within an input sequence
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

    # Calculate the relevant data for each frame, either bone length or velocities
    # frameBones = bone_lengths(leftSellerJoints, rightSellerJoints, buyerJoints, humanSkeleton)
    framePoses = velocities(leftSellerJoints, rightSellerJoints, buyerJoints)

    # Determine if super- or subsampling the data is necessary from user input
    if supersample >= 1:
        framePoses = super_sampling(framePoses, supersample)
    if subsample >= 1:
        framePoses = sub_sampling(framePoses, subsample)

    # Use the bone length data to filter out outliers and excessively noisy data
    segments = get_segments(framePoses)

    # Create a new copy of the data, to be reused for each data segment
    new_pickle = copy.deepcopy(group)

    # Write new pickle files, each one containing data for exactly one good segment of video
    partCount = 0
    for seg in segments:
        # Record data for each subject in the video
        for subject in range(0, len(group["subjects"])):
            # Reset sizes of the 2d numpy arrays in the new pickle dictionary
            new_subject = new_pickle["subjects"][subject]
            new_subject["joints19"] = np.zeros((57, seg[1] - seg[0]))

            # NOTE: Currently we discard the bodyNormal, faceNormal, and scores data. This section is deprecated
            # new_subject["bodyNormal"] = np.zeros((3, seg[1] - seg[0]))
            # new_subject["faceNormal"] = np.zeros((3, seg[1] - seg[0]))
            # new_subject["scores"] = np.zeros((19, seg[1] - seg[0]))
            # for j in range(0, 3):
            #     new_subject["bodyNormal"][j] = old_subject["bodyNormal"][j][seg[0]:seg[1]]
            #     new_subject["faceNormal"][j] = old_subject["faceNormal"][j][seg[0]:seg[1]]
            # for j in range(0, 19):
            #     new_subject["scores"][j] = old_subject["scores"][j][seg[0]:seg[1]]

            # Use trimmed segments from original data for all frame-dependent information
            for joint in range(0, 19):
                for frame in range(seg[1] - seg[0]):
                    # Assign the x-y-z coordinates individually, due to differing data structures
                    new_subject['joints19'][joint * 3][frame] = (np.array(framePoses)[frame, subject, joint, 0])
                    new_subject['joints19'][joint * 3 + 1][frame] = (np.array(framePoses)[frame, subject, joint, 1])
                    new_subject['joints19'][joint * 3 + 2][frame] = (np.array(framePoses)[frame, subject, joint, 2])

        # Write the new data to the designated files
        base = os.path.basename(inp)
        outFile = open(("%s/%s_part%d.pkl" % (output, base[:-4], partCount)), "wb")
        print("Writing data from frames %d:%d to file %s_part%d.pkl" % (seg[0], seg[1], base[:-4], partCount))
        pickle.dump(new_pickle, outFile)
        outFile.close()
        partCount += 1


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
        preprocess_frames(target, FLAGS.output, FLAGS.subsample, FLAGS.supersample)


if __name__ == "__main__":
    app.run(main)
