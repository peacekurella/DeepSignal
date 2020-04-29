import csv
import numpy as np
import os
import sys
import tensorflow as tf
from absl import flags
from absl import app
from pathlib import Path

sys.path.append('..')
import dataUtils.getData as gd
import dataUtils.standardize as strd
import dataUtils.dataVis as dv
import dataUtils.getTrajectory as traj

# set up flags
FLAGS = flags.FLAGS

def main(argv):
    """
	please change sequence length to a multiple of 3 in createRecords.py
    """

    if not (os.path.isdir(FLAGS.trainData)):
        print("Invalid directory for train data")
        sys.exit()                                                                 

    tf.compat.v1.enable_eager_execution()
    # read the files
    files = list(map(lambda x: os.path.join(FLAGS.trainData, x), os.listdir(FLAGS.trainData)))
    dataset = tf.data.TFRecordDataset(files)

    # parse the examples
    dataset = dataset.map(gd.parse_example)

    # deserialize the tensors
    dataset = dataset.map(gd.deserialize_example)

    dataset = dataset.take(1)

    for b, l, r in dataset:

	    tb, jb = b

	    tl, jl = l

	    tr, jr = r


	    nb = strd.standardize(jb)

	    nb = nb.numpy()

	    db = strd.destandardize(nb)


	    nl = strd.standardize(jl)

	    nl = nl.numpy()

	    dl = strd.destandardize(nl)


	    nr = strd.standardize(jr)

	    nr = nr.numpy()

	    dr = strd.destandardize(nr)

	    print(tb.numpy().shape)

	    buyer = traj.convert_to_absolute(tb.numpy(), jb.numpy())

	    left = traj.convert_to_absolute(tl.numpy(), jl.numpy())

	    right = traj.convert_to_absolute(tr.numpy(), jr.numpy())

	    dv.create_animation(buyer.T, left.T, right.T, None, "./input")


	    buyer = traj.convert_to_absolute(tb.numpy(), db)

	    left = traj.convert_to_absolute(tl.numpy(), dl)

	    right = traj.convert_to_absolute(tr.numpy(), dr)

	    dv.create_animation(buyer.T, left.T, right.T, None, "./output")

if __name__ == '__main__':
    app.run(main)
