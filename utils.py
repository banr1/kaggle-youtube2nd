import numpy as np
import tensorflow as tf
from tensorflow import logging

def dequantize(feat_vector, max_quantized_value=2, min_quantized_value=-2):
    assert max_quantized_value > min_quantized_value
    quantized_range = max_quantized_value - min_quantized_value
    scalar = quantized_range / 255.0
    bias = (quantized_range / 512.0) + min_quantized_value
    return feat_vector * scalar + bias

def make_summary(name, value):
    summary = tf.Summary()
    val = summary.value.add()
    val.tag = str(name)
    val.simple_value = float(value)
    return summary

def get_feature_names_and_sizes(feature_names, feature_sizes):
    name_list = [name.strip() for name in feature_names.split(',')]
    size_list = [int(size) for size in feature_sizes.split(',')]
    if len(name_list) != len(size_list):
        logging.error(f"length of the feature names (={len(name_list)}) != "\
                      f"length of feature sizes (={len(size_list)})")
    return name_list, size_list

def get_input_data_tensors(reader,
                           data_pattern,
                           shuffle,
                           batch_size=1024,
                           num_readers=1,
                           num_epochs=None,
                           phase=""):
    logging.info(f"Using batch size of {batch_size} for {phase}.")
    with tf.name_scope(f"{phase}_input"):
        files = tf.gfile.Glob(data_pattern)
        if not files:
            raise IOError(f"Unable to find {phase} files. "\
                          f"data_pattern='{data_pattern}'.")
        logging.info(f"Number of {phase} files: {len(files)}.")
        filename_queue = tf.train.string_input_producer(files,
                                                        num_epochs=num_epochs,
                                                        shuffle=shuffle)
        phase_data = [reader.prepare_reader(filename_queue)
                      for _ in range(num_readers)]
        if shuffle:
            return tf.train.shuffle_batch_join(phase_data,
                                               batch_size=batch_size,
                                               capacity=batch_size*5,
                                               min_after_dequeue=batch_size,
                                               allow_smaller_final_batch=True,
                                               enqueue_many=True)
        else:
            return tf.train.batch_join(phase_data,
                                       batch_size=batch_size,
                                       allow_smaller_final_batch=True,
                                       enqueue_many=True)
