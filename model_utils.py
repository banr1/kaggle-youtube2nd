import tensorflow as tf

def sample_random_sequence(model_input, num_frames, num_samples):
    batch_size = tf.shape(model_input)[0]
    frame_index_offset = tf.tile(
        tf.expand_dims(tf.range(num_samples), 0), [batch_size, 1])
    max_start_frame_index = tf.maximum(num_frames - num_samples, 0)
    start_frame_index = tf.cast(
        tf.multiply(tf.random_uniform([batch_size, 1]), tf.cast(max_start_frame_index + 1, tf.float32)), tf.int32)
    frame_index = tf.minimum(start_frame_index + frame_index_offset, tf.cast(num_frames - 1, tf.int32))
    batch_index = tf.tile(tf.expand_dims(tf.range(batch_size), 1), [1, num_samples])
    index = tf.stack([batch_index, frame_index], 2)
    return tf.gather_nd(model_input, index)

def sample_random_frames(model_input, num_frames, num_samples):
    batch_size = tf.shape(model_input)[0]
    frame_index = tf.cast(tf.multiply(tf.random_uniform([batch_size, num_samples]),
                                      tf.tile(tf.cast(num_frames, tf.float32), [1, num_samples])), tf.int32)
    batch_index = tf.tile(tf.expand_dims(tf.range(batch_size), 1), [1, num_samples])
    index = tf.stack([batch_index, frame_index], 2)
    return tf.gather_nd(model_input, index)

def frame_pooling(frames, method, **unused_params):
    if method == "average":
        return tf.reduce_mean(frames, 1)
    elif method == "max":
        return tf.reduce_max(frames, 1)
    elif method == "none":
        feature_size = frames.shape_as_list()[2]
        return tf.reshape(frames, [-1, feature_size])
    else:
        raise ValueError(f"Unrecognized pooling method: {method}")
