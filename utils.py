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
