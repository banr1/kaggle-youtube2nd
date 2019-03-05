import numpy as np
import time
import tensorflow as tf
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging
FLAGS = flags.FLAGS

from readers import get_reader
from utils import make_summary, get_input_data_tensors

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
    flags.DEFINE_integer("check_point", -1, "")

def format_lines(video_ids, predictions, top_k):
    batch_size = len(video_ids)
    for video_index in range(batch_size):
        top_indices = np.argpartition(predictions[video_index], -top_k)[-top_k:]
        line = [(class_index, predictions[video_index][class_index])
                for class_index in top_indices]
        line = sorted(line, key=lambda p: -p[1])
        yield video_ids[video_index].decode('utf-8') + "," \
              + " ".join(f"{label} {confidence}" for label,confidence in line) \
              + "\n"

def inference(reader, train_dir, test_data_pattern,
              out_file_location, batch_size, top_k):
    with tf.Session() as sess, gfile.Open(out_file_location, "w+") as out_file:
        video_id_batch, video_batch, unused_labels, num_frames_batch = \
                get_input_data_tensors(reader,
                                       test_data_pattern,
                                       shuffle=False,
                                       batch_size=batch_size,
                                       num_readers=1,
                                       num_epochs=1,
                                       phase="test")
        latest_checkpoint = tf.train.latest_checkpoint(train_dir)
        if latest_checkpoint is None:
            raise Exception("unable to find a checkpoint "\
                            f"at location: {train_dir}")
        else:
            if FLAGS.check_point < 0:
                meta_graph_location = latest_checkpoint + ".meta"
            else:
                meta_graph_location = FLAGS.train_dir + \
                                      f"/model.ckpt-f{FLAGS.check_point}.meta"
                latest_checkpoint = FLAGS.train_dir + \
                                    f"/model.ckpt-{FLAGS.check_point}"
            logging.info("loading meta-graph: " + meta_graph_location)
        saver = tf.train.import_meta_graph(meta_graph_location,
                                           clear_devices=True)
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

        sess.run(set_up_init_ops(tf.get_collection_ref(
                tf.GraphKeys.LOCAL_VARIABLES)))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        num_examples_processed = 0
        start_time = time.time()
        out_file.write("VideoId,LabelConfidencePairs\n")
        try:
            while not coord.should_stop():
                video_id_batch_, video_batch_, num_frames_batch_ = sess.run(
                        [video_id_batch, video_batch, num_frames_batch])
                predictions_, = sess.run(
                        [predictions_tensor],
                        feed_dict={input_tensor: video_batch_,
                                   num_frames_tensor: num_frames_batch_})
                now = time.time()
                num_examples_processed += len(video_batch_)
                num_classes = predictions_.shape[1]
                logging.info(
                        f"num examples processed: {num_examples_processed} "\
                        f"elapsed seconds: {now-start_time:.2f}")
                for line in format_lines(video_id_batch_, predictions_, top_k):
                    out_file.write(line)
                out_file.flush()
        except tf.errors.OutOfRangeError:
            logging.info("Done with inference. "\
                         f"The output file was written to {out_file_location}")
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()

def main(unused_argv):
    logging.set_verbosity(tf.logging.INFO)
    reader = get_reader(FLAGS.feature_names,
                        FLAGS.feature_sizes,
                        FLAGS.frame_features)
    if FLAGS.output_file is "":
        raise ValueError("'output_file' was not specified. "\
                         "Unable to continue with inference.")
    if FLAGS.test_data_pattern is "":
        raise ValueError("'test_data_pattern' was not specified. "\
                         "Unable to continue with inference.")
    inference(reader, FLAGS.train_dir, FLAGS.test_data_pattern,
              FLAGS.output_file, FLAGS.batch_size, FLAGS.top_k)

if __name__ == "__main__":
    tf.app.run()
