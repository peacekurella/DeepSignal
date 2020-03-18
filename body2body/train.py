import tensorflow as tf
import sys
import modelBuilder as model

# add directories to path for module discovery
sys.path.insert(0, "haggling")
sys.path.insert(0, "haggling/body2body")

def parse_example(example_proto):
    """
    Parses examples from the record files
    :param example_proto: input example proto_buffer string
    :return: parsed example
    """
    # create a feature descriptor
    feature_description = {
        'br': tf.io.FixedLenFeature([], tf.string),
        'ls': tf.io.FixedLenFeature([], tf.string),
        'rs': tf.io.FixedLenFeature([], tf.string)
    }
    return tf.io.parse_single_example(example_proto, feature_description)

def deserialize_example(example):
    """
    Deserializes the tensors in parsed examples
    :param example: input example to be parsed
    :return: (buyerJoints, leftSellerJoints, rightSellerJoints) tuple containing the sequences
    """
    # cast to float32 for better performance
    buyerJoints = tf.cast(tf.io.parse_tensor(example['br'], out_type=tf.double), tf.float32)
    leftSellerJoints = tf.cast(tf.io.parse_tensor(example['ls'], out_type=tf.double), tf.float32)
    rightSellerJoints = tf.cast(tf.io.parse_tensor(example['rs'], out_type=tf.double), tf.float32)
    return buyerJoints, leftSellerJoints, rightSellerJoints

filename = "/home/prashanth/Desktop/haggling/records/170221_haggling_b1_group0.TFrecord"
files = [filename]
dataset = tf.data.TFRecordDataset(files)
dataset = dataset.map(parse_example)
dataset = dataset.map(deserialize_example)
dataset = dataset.batch(64)

for x, y, z in dataset.take(1):
    input = tf.concat([x, y, z], axis=2)
    encoder = model.Encoder(1024, 64, 2, 0.2)
    decoder = model.Decoder(57, 1024, 64, 0, 0.2)
    hidden = encoder.initialize_hidden_state()
    print(hidden.shape)
    output, state = encoder(input, hidden, True)
    output, state, attention_weights = decoder(z[:, 0, :], state, output, True)
    print(output, attention_weights)