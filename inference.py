import numpy as np
import time
import tensorflow as tf
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging
FLAGS = flags.FLAGS

from readers import YT8MFrameFeatureReader, YT8MAggregatedFeatureReader
from utils import make_summary, get_feature_names_and_sizes

if __name__ == '__main__':
    flags.DEFINE_string("train_dir", "../log/", "")
    flags.DEFINE_string("test_data_pattern", "../input/test/*.tfrecord", "")
    flags.DEFINE_string("feature_names", "rgb,audio", "")
    flags.DEFINE_string("feature_sizes", "1024,128", "")
    flags.DEFINE_bool("frame_features", True, "")
    flags.DEFINE_integer("batch_size", 8192, "")
    flags.DEFINE_integer("num_readers", 4, "")
    flags.DEFINE_string("output_file", "", "")
    flags.DEFINE_integer("top_k", 20, "")
    flags.DEFINE_integer("check_point",-1, "")

def format_lines(video_ids, predictions, top_k):
    batch_size = len(video_ids)
    for video_index in range(batch_size):
        top_indices = np.argpartition(predictions[video_index], -top_k)[-top_k:]
        line = [(class_index, predictions[video_index][class_index]) for class_index in top_indices]
        line = sorted(line, key=lambda p: -p[1])
        yield video_ids[video_index].decode('utf-8') + "," + " ".join("%i %f" % pair for pair in line) + "\n"

def get_input_data_tensors(reader, data_pattern, batch_size, num_readers=1):
    with tf.name_scope("input"):
        files = gfile.Glob(data_pattern)
        if not files:
            raise IOError("Unable to find input files. data_pattern='" + data_pattern + "'")
        logging.info("number of input files: " + str(len(files)))
        filename_queue = tf.train.string_input_producer(files, num_epochs=1, shuffle=False)
        examples_and_labels = [reader.prepare_reader(filename_queue) for _ in range(num_readers)]
        video_id_batch, video_batch, unused_labels, num_frames_batch = (
                tf.train.batch_join(examples_and_labels,
                                    batch_size=batch_size,
                                    allow_smaller_final_batch = True,
                                    enqueue_many=True))
        return video_id_batch, video_batch, num_frames_batch

def inference(reader, train_dir, data_pattern, out_file_location, batch_size, top_k):
    with tf.Session() as sess, gfile.Open(out_file_location, "w+") as out_file:
        video_id_batch, video_batch, num_frames_batch = get_input_data_tensors(reader, data_pattern, batch_size)
        latest_checkpoint = tf.train.latest_checkpoint(train_dir)
        if latest_checkpoint is None:
            raise Exception("unable to find a checkpoint at location: %s" % train_dir)
        else:
            if FLAGS.check_point < 0:
                meta_graph_location = latest_checkpoint + ".meta"
            else:
                meta_graph_location = FLAGS.train_dir + "/model.ckpt-" + str(FLAGS.check_point) + ".meta"
                latest_checkpoint = FLAGS.train_dir + "/model.ckpt-" + str(FLAGS.check_point)
            logging.info("loading meta-graph: " + meta_graph_location)
        saver = tf.train.import_meta_graph(meta_graph_location, clear_devices=True)
        logging.info("restoring variables from " + latest_checkpoint)
        saver.restore(sess, latest_checkpoint)
        input_tensor = tf.get_collection("input_batch_raw")[0]
        num_frames_tensor = tf.get_collection("num_frames")[0]
        predictions_tensor = tf.get_collection("predictions")[0]
        def set_up_init_ops(variables):
            init_op_list = []
            for variable in list(variables):
                if "train_input" in variable.name:
                    init_op_list.append(tf.assign(variable, 1))
                    variables.remove(variable)
            init_op_list.append(tf.variables_initializer(variables))
            return init_op_list

        sess.run(set_up_init_ops(tf.get_collection_ref(tf.GraphKeys.LOCAL_VARIABLES)))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        num_examples_processed = 0
        start_time = time.time()
        out_file.write("VideoId,LabelConfidencePairs\n")
        try:
            while not coord.should_stop():
                video_id_batch_val, video_batch_val,num_frames_batch_val = sess.run([video_id_batch, video_batch, num_frames_batch])
                predictions_val, = sess.run([predictions_tensor],
                                            feed_dict={input_tensor: video_batch_val, num_frames_tensor: num_frames_batch_val})
                now = time.time()
                num_examples_processed += len(video_batch_val)
                num_classes = predictions_val.shape[1]
                logging.info("num examples processed: " + str(num_examples_processed) +
                             " elapsed seconds: " + "{0:.2f}".format(now-start_time))
                for line in format_lines(video_id_batch_val, predictions_val, top_k):
                    out_file.write(line)
                out_file.flush()
        except tf.errors.OutOfRangeError:
            logging.info('Done with inference. The output file was written to ' + out_file_location)
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()

def main(unused_argv):
    logging.set_verbosity(tf.logging.INFO)
    feature_names, feature_sizes = get_feature_names_and_sizes(FLAGS.feature_names, FLAGS.feature_sizes)
    if FLAGS.frame_features:
        reader = YT8MFrameFeatureReader(feature_names=feature_names, feature_sizes=feature_sizes)
    else:
        reader = YT8MAggregatedFeatureReader(feature_names=feature_names, feature_sizes=feature_sizes)
    if FLAGS.output_file is "":
        raise ValueError("'output_file' was not specified. Unable to continue with inference.")
    if FLAGS.test_data_pattern is "":
        raise ValueError("'test_data_pattern' was not specified. Unable to continue with inference.")
    inference(reader, FLAGS.train_dir, FLAGS.test_data_pattern,
        FLAGS.output_file, FLAGS.batch_size, FLAGS.top_k)

if __name__ == "__main__":
    tf.app.run()
