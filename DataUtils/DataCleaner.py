import copy
import math
import os
import pickle
import sys
import numpy as np


def super_sampling(inp, k):
    """
    Returns a linearly interpolated supersample of the original sequence
    :param inp: Input sequence of positions
    :param k: Number of interpolated frames, should be an integer >= 1
    :return: The newly generated sequence of (k + 1) * n frames
    """
    inp = np.array(inp)
    out = []
    for frame in range(1, len(inp)):
        diff = inp[frame] - inp[frame - 1]  # Find the average change in position between frames
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
    for frame in range(0, len(inp), k):
        out.append(inp[frame])
    return np.array(out)


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


def bone_lengths(lSJoints, rSJoints, bJoints, humanSkeleton, supersample=0, subsample=0):
    """
    Calculate and collect the float length of each bone into a single list.
    :param lSJoints: Raw joint data for the left seller
    :param rSJoints: Raw joint data for the right seller
    :param bJoints: Raw joint data for the buyer
    :param humanSkeleton: List of joint pairs, representing the joint connections (bones) in our person models.
    :param supersample: How many frames to linearly interpolate between original frame data
    (i.e. 1, 2, 3, ... extra frames between each original frame)
    :param subsample: How many frames to skip to subsample the data
    :return: A list of length triples, representing the lengths of each person"s bones. Example format is:
    Frame k: [leftBone1, rightBone1, buyerBone1, leftBone2, rightBone2, buyerBone2, leftBone3, ...]
    """
    # Determine the positions of each person"s joints
    leftSellerFrameJoints, rightSellerFrameJoints, buyerFrameJoints = joint_positions(lSJoints, rSJoints, bJoints)

    # Check for super-/subsampling requirements before finding problem frames
    if supersample != 0:
        leftSellerFrameJoints = super_sampling(leftSellerFrameJoints, supersample)
        rightSellerFrameJoints = super_sampling(rightSellerFrameJoints, supersample)
        buyerFrameJoints = super_sampling(buyerFrameJoints, supersample)
    if subsample != 0:
        leftSellerFrameJoints = sub_sampling(leftSellerFrameJoints, subsample)
        rightSellerFrameJoints = sub_sampling(rightSellerFrameJoints, subsample)
        buyerFrameJoints = sub_sampling(buyerFrameJoints, subsample)

    # Collect frames where all of a joints coords are at 0: Avoid those frames
    problem_frames = []
    for frame in range(0, len(leftSellerFrameJoints)):
        for joint in range(0, 19):
            ls = leftSellerFrameJoints[frame][joint]
            rs = rightSellerFrameJoints[frame][joint]
            b = buyerFrameJoints[frame][joint]
            if (ls[0] == 0 and ls[1] == 0 and ls[2] == 0) or (ls[0] == 0 and rs[1] == 0 and rs[2] == 0) or (
                    b[0] == 0 and b[1] == 0 and b[2] == 0):
                problem_frames.append(frame)
                continue

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

    return frameBones, problem_frames


def get_segments(bones, min_length, std_devs=3):
    """
    Determines which segments of a video"s data are of acceptable quality and length.
    :param bones: Input bone length data for the video"s frames.
    :param min_length: Minimum length required for acceptable clips of data
    :param std_devs: How many standard deviations to use in noise tolerance
    :return: List of pairs, describing the start and end frames of each acceptable segment.
    """
    # Within each frame, check if the bone lengths fall inside our tolerance threshold
    # If not, add the frame to a list of bad frames for later use
    problemFrames = []
    frameBones = np.array(bones)  # Convert to numpy array find distribution stats easier

    # Current method uses mean and standard deviations for the error tolerances
    means = np.mean(frameBones, axis=0)
    tolerance = np.std(frameBones, axis=0) * std_devs

    # Alternative method: Use percentiles as thresholds
    # upper_percent = np.percentile(frameBones, 99, axis=0)
    # lower_percent = np.percentile(frameBones, 1, axis=0)

    for frame in range(0, len(frameBones)):
        for joint in range(0, len(frameBones[frame])):
            lower_thresh = means[joint] - tolerance[joint]
            upper_thresh = means[joint] + tolerance[joint]
            # upper_thresh = upper_percent[joint]
            # lower_thresh = lower_percent[joint]
            if np.any(frameBones[frame][joint] < lower_thresh) or np.any(frameBones[frame][joint] > upper_thresh):
                problemFrames.append(frame)
                break

    # Define partitions for writing out files as the start and end frames numbers of whole segments.
    segments = []
    for target in range(len(problemFrames) - 1):
        # If a sequence is more than the desired length, add the trimmed version to those accepted.
        if problemFrames[target + 1] - problemFrames[target] > min_length:
            segments.append((problemFrames[target] + 1, problemFrames[target + 1]))

    # If no problem frames found, write all of the video's data out
    if len(problemFrames) == 0:
        segments.append((0, len(frameBones) - 1))

    return segments


class DataCleaner:
    """
    Cleans the data to eliminate outliers and otherwise noisy data
    """
    
    def __init__(self, input='../input', output='../preprocessed', subsample=0, supersample=0, min_length=50):
        """
        Initializes the DataCleaner
        """
        # Define the relevant connections ("bones") between the joints to visualize
        self.humanSkeleton = [
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
        self.input = input
        self.output = output
        self.subsample = subsample
        self.supersample = supersample
        self.min_length = min_length
        pass

    def preprocess_frames(self, inp):
        """
        Given some "raw" video data as a pickle file, produces some number of files containing the frame data for
        exactly one acceptable segment of video. Number of files produced depends upon the quality and quantity of the
        original"s video data.
        :param: inp: Input pickle file containing frame-by-frame position data
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

        for subject in group["subjects"]:
            if subject["humanId"] == leftSellerId:
                leftSellerJoints = subject["joints19"]
            if subject["humanId"] == rightSellerId:
                rightSellerJoints = subject["joints19"]
            if subject["humanId"] == buyerId:
                buyerJoints = subject["joints19"]

        # Calculate the bone lengths and problematic frames for the given data.
        frameBones, zero_frames = bone_lengths(leftSellerJoints, rightSellerJoints, buyerJoints, self.humanSkeleton,
                                               self.supersample, self.subsample)

        # Use the bone length data to filter out outliers and excessively noisy data
        bone_segments = get_segments(frameBones, self.min_length)

        # Merge the information to one list of usable segments
        zero_ptr = 0
        bone_ptr = 0
        while zero_ptr < len(zero_frames) and bone_ptr < len(bone_segments):
            z_frame = zero_frames[zero_ptr]
            if z_frame > bone_segments[bone_ptr][0]:
                if z_frame < bone_segments[bone_ptr][1]:
                    del (bone_segments[bone_ptr])
                else:
                    bone_ptr += 1
            else:
                zero_ptr += 1

        # Create a new copy of the data, to be reused for each data segment
        new_pickle = copy.deepcopy(group)

        # Write new pickle files, each one containing data for exactly one good segment of video
        partCount = 0
        for seg in bone_segments:
            # Record data for each subject in the video
            for subject in range(0, len(group["subjects"])):
                # Reset sizes of the 2d numpy arrays in the new pickle dictionary
                new_subject = new_pickle["subjects"][subject]

                # NOTE: Currently we don't update the bodyNormal, faceNormal, and scores data.
                # Use trimmed segments from original data for all frame-dependent positions
                new_subject["joints19"] = np.zeros((57, seg[1] - seg[0]))

                # Assign the clipped values for each separate pkl file.
                for joint in range(0, 57):
                    old_joints = group['subjects'][subject]['joints19'][joint]

                    # If supersampling, then we need to linearly interpolate data
                    if self.supersample != 0:
                        old_joints = super_sampling(old_joints, self.supersample)

                    new_subject['joints19'][joint] = old_joints[seg[0]:seg[1]]

            # Write the new data to the designated files
            base = os.path.basename(inp)
            outFile = open(("%s/%s_part%d.pkl" % (self.output, base[:-4], partCount)), "wb")
            print("Writing data from frames %d:%d to file %s_part%d.pkl" % (seg[0], seg[1], base[:-4], partCount))
            pickle.dump(new_pickle, outFile)
            outFile.close()
            partCount += 1

    def clean(self):
        # Check if the input directory exists
        if not self.input:
            print("Invalid input pkl directory")
            sys.exit()

        # Create the output directory if none exist
        if not (os.path.isdir(self.output)):
            try:
                os.mkdir(self.output)
            except:
                print("Error creating output directory")
                sys.exit()

        # For each pickle file within the input directory, preprocess its data
        for file in os.listdir(self.input):
            target = "%s/%s" % (self.input, file)
            print("Reading data from: ", target)
            self.preprocess_frames(target)
