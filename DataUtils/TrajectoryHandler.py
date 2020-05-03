import numpy as np
import tensorflow as tf


def absolute_to_relative(prev_state, cur_state):
    """
    Returns relative trajectory from prev_state and cur_state
    :param prev_state: absolute previous position
    :param cur_state: absolute current position
    :return: relative current position
    """
    # initialize the roots
    prev_state = tf.squeeze(prev_state)
    cur_state = tf.squeeze(cur_state)
    previous_points = tf.split(prev_state, axis=0, num_or_size_splits=19)
    current_points = tf.split(cur_state, axis=0, num_or_size_splits=19)

    root_previous = previous_points[2]
    root_current = current_points[2]

    cur_skeleton = []

    # go through the whole skeleton
    for i, point in enumerate(current_points):

        # convert into translations from the root
        if not i == 2:
            cur_skeleton.append(
                point - root_current
            )
        else:
            cur_skeleton.append(
                root_current - root_previous
            )

    return tf.expand_dims(tf.concat(cur_skeleton, axis=0), axis=0)


def convert_to_trajectory(input_sequence):
    """
    Converts an input absolute sequence to trajectories
    :param input_sequence: absolute input sequence to be converted to relative trajectories
    :return: absolute start position and relative trajectory
    """

    # start state is first position of the sequence
    time_steps = tf.split(input_sequence, axis=0, num_or_size_splits=input_sequence.shape[0])
    start = time_steps[0]
    prev_state = time_steps[0]

    # final sequence
    output_sequence = []

    for i in time_steps:
        cur_state = i
        output_sequence.append(absolute_to_relative(prev_state, cur_state))
        prev_state = i

    return start, tf.concat(output_sequence, axis=0)


def relative_to_absolute(prev_state, cur_state):
    """
    Returns absolute positions from prev_state and cur_state
    :param prev_state: absolute previous position
    :param cur_state: relative current position
    :return: absolute current position
    """

    # find the current root position
    root_previous = np.array([prev_state[6], prev_state[7], prev_state[8]])
    root_current = np.array([cur_state[6], cur_state[7], cur_state[8]])
    root_current = root_current + root_previous

    # final skeleton
    cur_skeleton = []

    for i in range(0, len(prev_state), 3):

        # convert into absolute postions
        if not i == 2:
            cur_skeleton.extend([
                cur_state[i] + root_current[0],
                cur_state[i + 1] + root_current[1],
                cur_state[i + 2] + root_current[2],
            ])
        else:
            cur_skeleton.extend([
                root_current[0],
                root_current[1],
                root_current[2]
            ])

    return cur_skeleton


def convert_to_absolute(start, trajectory):
    """
    Converts trajectories into absolute positions
    :param start: absolute start position of the subject
    :param trajectory: trajectory of the subject
    :return: output_sequence
    """

    # first state is start
    prev_state = np.squeeze(start)

    # final sequence
    output_sequence = []

    # convert the trajectory back into absolute positions
    for i in range(trajectory.shape[0]):
        cur_state = np.squeeze(trajectory[i])
        output_sequence.append([relative_to_absolute(prev_state, cur_state)])
        prev_state = np.squeeze(output_sequence[i])

    return np.squeeze(np.array(output_sequence))
