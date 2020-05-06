from absl import app
from absl import flags
from DataUtils.RecordWriter import RecordWriter
from DataUtils.InputStandardizer import generate_stats
from DataUtils.DataCleaner import DataCleaner

# set up flags
FLAGS = flags.FLAGS

flags.DEFINE_string('input', 'Data/input', 'Input Directory')
flags.DEFINE_string('preprocessed', 'Data/preprocessed', 'preprocessed pkl directory')
flags.DEFINE_string('train', 'Data/train', 'Train record output directory')
flags.DEFINE_string('test', 'Data/test', 'Train record output directory')
flags.DEFINE_integer('seqLength', 80, 'Sequence length')
flags.DEFINE_integer('testFiles', 50, 'Maximum Number of pkl files used for testing')
flags.DEFINE_integer('keypoints', 57, 'Number of keypoints in the skeleton')

flags.DEFINE_string('stats', 'Data/stats', 'Directory to store stats')


def main(argv):
    """
    Handles the data pre-processing pipeline
    :param argv:
    :return:
    """
    # preprocess the data using Data Cleaner, dividing into the training and testing data
    cleaner = DataCleaner(input=FLAGS.input, output=FLAGS.preprocessed, min_length=FLAGS.seqLength)
    cleaner.clean()

    # create the record writer
    writer = RecordWriter(
        FLAGS.preprocessed,
        FLAGS.train,
        FLAGS.test,
        FLAGS.testFiles,
        FLAGS.keypoints,
        FLAGS.seqLength
    )

    # create the tf records
    writer.create_records()

    # generate the statistics
    generate_stats(
        FLAGS.train,
        FLAGS.stats
    )


if __name__ == "__main__":
    app.run(main)
