import os
import json
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging
FLAGS = flags.FLAGS

import losses
import frame_level_models
import video_level_models
from eval_utils import calculate_hit_at_one, calculate_perr, calculate_gap
from export_model import ModelExporter
from readers import YT8MFrameFeatureReader, YT8MAggregatedFeatureReader
from utils import make_summary, get_feature_names_and_sizes

if __name__ == "__main__":
    flags.DEFINE_string("train_dir", "../log/", "")
    flags.DEFINE_string("train_data_pattern", "../input/train/*.tfrecord", "")
    flags.DEFINE_string("feature_names", "rgb,audio", "")
    flags.DEFINE_string("feature_sizes", "1024,128", "")
    flags.DEFINE_bool("frame_features", True, "")
    flags.DEFINE_integer("batch_size", 1024, "")
    flags.DEFINE_integer("num_readers", 8, "")
    flags.DEFINE_string("model", "LogisticModel", "")
    flags.DEFINE_bool("start_new_model", False, "")
    flags.DEFINE_string("label_loss", "CrossEntropyLoss", "")
    flags.DEFINE_float("regularization_penalty", 1, "")
    flags.DEFINE_float("base_lr", 0.001, "")
    flags.DEFINE_float("lr_decay", 0.9, "")
    flags.DEFINE_float("lr_decay_examples", 4000000, "")
    flags.DEFINE_integer("num_epochs", 15, "")
    flags.DEFINE_integer("max_steps", None, "")
    flags.DEFINE_integer("export_model_steps", 10000, "")
    flags.DEFINE_string("optimizer", "AdamOptimizer", "")
    flags.DEFINE_float("clip_gradient_norm", 1.0, "")
    flags.DEFINE_bool("log_device_placement", False, "")

def validate_class_name(flag_value, category, modules, expected_superclass):
    candidates = [getattr(module, flag_value, None) for module in modules]
    for candidate in candidates:
        if not candidate:
            continue
        if not issubclass(candidate, expected_superclass):
            raise flags.FlagsError(
                    f"{category} '{flag_value}' doesn't inherit "\
                    f"from {expected_superclass.__name__}.")
        return True
    raise flags.FlagsError(f"Unable to find {category} '{flag_value}'.")

def get_input_data_tensors(reader,
                           data_pattern,
                           batch_size=1024,
                           num_epochs=None,
                           num_readers=1):
    logging.info(f"Using batch size of {batch_size} for training.")
    with tf.name_scope("train_input"):
        files = gfile.Glob(data_pattern)
        if not files:
            raise IOError("Unable to find training files. "\
                          f"data_pattern='{data_pattern}'.")
        logging.info(f"Number of training files: {len(files)}.")
        filename_queue = tf.train.string_input_producer(files,
                                                        num_epochs=num_epochs,
                                                        shuffle=True)
        training_data = [reader.prepare_reader(filename_queue)
                          for _ in range(num_readers)]
        return tf.train.shuffle_batch_join(training_data,
                                           batch_size=batch_size,
                                           capacity=batch_size*5,
                                           min_after_dequeue=batch_size,
                                           allow_smaller_final_batch=True,
                                           enqueue_many=True)

def find_class_by_name(name, modules):
    modules = [getattr(module, name, None) for module in modules]
    return next(a for a in modules if a)

def build_graph(reader,
                model,
                train_data_pattern,
                label_loss_fn=losses.CrossEntropyLoss(),
                batch_size=1024,
                base_learning_rate=0.01,
                learning_rate_decay_examples=1000000,
                learning_rate_decay=0.95,
                optimizer_class=tf.train.AdamOptimizer,
                clip_gradient_norm=1.0,
                regularization_penalty=1,
                num_readers=1,
                num_epochs=None):
    global_step = tf.Variable(0, trainable=False, name="global_step")
    learning_rate = tf.train.exponential_decay(base_learning_rate,
                                               global_step * batch_size,
                                               learning_rate_decay_examples,
                                               learning_rate_decay,
                                               staircase=True)
    tf.summary.scalar("learning_rate", learning_rate)
    optimizer = optimizer_class(learning_rate)
    unused_video_id, model_input_raw, labels_batch, num_frames = \
            get_input_data_tensors(reader,
                                   train_data_pattern,
                                   batch_size=batch_size,
                                   num_readers=num_readers,
                                   num_epochs=num_epochs)
    tf.summary.histogram("model/input_raw", model_input_raw)
    feature_dim = len(model_input_raw.get_shape()) - 1
    model_input = tf.nn.l2_normalize(model_input_raw, feature_dim)
    with tf.name_scope("model"):
        result = model.create_model(model_input,
                                    num_frames=num_frames,
                                    vocab_size=reader.num_classes,
                                    labels=labels_batch)
        for variable in slim.get_model_variables():
            tf.summary.histogram(variable.op.name, variable)
        predictions = result["predictions"]
        if "loss" in result.keys():
            label_loss = result["loss"]
        else:
            label_loss = label_loss_fn.calculate_loss(predictions, labels_batch)
        tf.summary.scalar("label_loss", label_loss)
        if "regularization_loss" in result.keys():
            reg_loss = result["regularization_loss"]
        else:
            reg_loss = tf.constant(0.0)
        reg_losses = tf.losses.get_regularization_losses()
        if reg_losses:
            reg_loss += tf.add_n(reg_losses)
        if regularization_penalty != 0:
            tf.summary.scalar("reg_loss", reg_loss)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if "update_ops" in result.keys():
            update_ops += result["update_ops"]
        if update_ops:
            with tf.control_dependencies(update_ops):
                barrier = tf.no_op(name="gradient_barrier")
                with tf.control_dependencies([barrier]):
                    label_loss = tf.identity(label_loss)
        final_loss = regularization_penalty * reg_loss + label_loss
        train_op = slim.learning.create_train_op(
                final_loss,
                optimizer,
                global_step=global_step,
                clip_gradient_norm=clip_gradient_norm)
        tf.add_to_collection("global_step", global_step)
        tf.add_to_collection("loss", label_loss)
        tf.add_to_collection("predictions", predictions)
        tf.add_to_collection("input_batch_raw", model_input_raw)
        tf.add_to_collection("input_batch", model_input)
        tf.add_to_collection("num_frames", num_frames)
        tf.add_to_collection("labels", tf.cast(labels_batch, tf.float32))
        tf.add_to_collection("train_op", train_op)

class Trainer(object):
    def __init__(self, cluster, task, train_dir, model, reader, model_exporter,
                 log_device_placement=True, export_model_steps=1000,
                 max_steps=None):
        self.cluster = cluster
        self.task = task
        self.is_master = (task.type == "master" and task.index == 0)
        self.train_dir = train_dir
        self.config = tf.ConfigProto(log_device_placement=log_device_placement)
        self.model = model
        self.reader = reader
        self.model_exporter = model_exporter
        self.max_steps = max_steps
        self.max_steps_reached = False
        self.export_model_steps = export_model_steps
        self.last_model_export_step = 0
        if self.is_master and self.task.index > 0:
            raise StandardError(f"{str_task(self.task)}: "\
                                "Only one replica of master expected")

    def run(self, start_new_model=False):
        if self.is_master and start_new_model:
            self.remove_training_directory(self.train_dir)
        target, device_fn = self.start_server_if_distributed()
        meta_filename = self.get_meta_filename(start_new_model, self.train_dir)
        with tf.Graph().as_default() as graph:
            if meta_filename:
                saver = self.recover_model(meta_filename)
            with tf.device(device_fn):
                if not meta_filename:
                    saver = self.build_model(self.model, self.reader)
                global_step = tf.get_collection("global_step")[0]
                loss = tf.get_collection("loss")[0]
                predictions = tf.get_collection("predictions")[0]
                labels = tf.get_collection("labels")[0]
                train_op = tf.get_collection("train_op")[0]
                init_op = tf.global_variables_initializer()
        sv = tf.train.Supervisor(graph,
                                 logdir=self.train_dir,
                                 init_op=init_op,
                                 is_chief=self.is_master,
                                 global_step=global_step,
                                 save_model_secs=15 * 60,
                                 save_summaries_secs=120,
                                 saver=saver)
        logging.info(f"{str_task(self.task)}: Starting managed session.")
        with sv.managed_session(target, config=self.config) as sess:
            try:
                logging.info(f"{str_task(self.task)}: Entering training loop.")
                while (not sv.should_stop()) and (not self.max_steps_reached):
                    batch_start_time = time.time()
                    _, global_step_, loss_, predictions_, labels_ = sess.run(
                            [train_op, global_step, loss, predictions, labels])
                    secs_per_batch = time.time() - batch_start_time
                    if self.max_steps and self.max_steps <= global_step_:
                        self.max_steps_reached = True
                    if self.is_master:
                        examples_per_sec = labels_.shape[0] / secs_per_batch
                        hit_at_one = calculate_hit_at_one(predictions_, labels_)
                        perr = calculate_perr(predictions_, labels_)
                        gap = calculate_gap(predictions_, labels_)
                        logging.info(f"{str_task(self.task)}: "\
                                     f"training step {global_step_}| "\
                                     f"Hit@1: {hit_at_one:.2f} "\
                                     f"PERR: {perr:.2f} "\
                                     f"GAP: {gap:.2f} "\
                                     f"Loss: {loss_:.5f}")
                        summary_dict = {
                            "model/Training_Hit@1": hit_at_one,
                            "model/Training_Perr": perr,
                            "model/Training_GAP": gap,
                            "global_step/Examples/Second": examples_per_sec,
                            }
                        for key, value in summary_dict.items():
                            sv.summary_writer.add_summary(
                                    make_summary(key, value), global_step_)
                        sv.summary_writer.flush()
                        time_to_export = \
                                ((self.last_model_export_step == 0) or
                                 (global_step_ - self.last_model_export_step \
                                  >= self.export_model_steps))
                        if self.is_master and time_to_export:
                            self.export_model(global_step_, sv, sess)
                            self.last_model_export_step = global_step_
                if self.is_master:
                    self.export_model(global_step_, sv, sess)
            except tf.errors.OutOfRangeError:
                logging.info(f"{str_task(self.task)}: "\
                             "Done training -- epoch limit reached.")
        logging.info(f"{str_task(self.task)}: Exited training loop.")
        sv.Stop()

    def export_model(self, global_step_val, sv, session):
        if global_step_val == self.last_model_export_step:
            return
        last_checkpoint = sv.saver.save(session, sv.save_path, global_step_val)
        model_dir = f"{self.train_dir}/export/step_{global_step_val}"
        logging.info(f"{str_task(self.task)}: Exporting the model "\
                     f"at step {global_step_val} to {model_dir}.")
        self.model_exporter.export_model(model_dir=model_dir,
                                         global_step_val=global_step_val,
                                         last_checkpoint=last_checkpoint)

    def start_server_if_distributed(self):
        if self.cluster:
            logging.info(f"{str_task(self.task)}: Starting trainer "\
                         f"within cluster {self.cluster.as_dict()}.")
            server = start_server(self.cluster, self.task)
            target = server.target
            device_fn = tf.train.replica_device_setter(
                    ps_device="/job:ps",
                    worker_device=f"/job:{self.task.type}/"\
                                  f"task:{self.task.index}",
                    cluster=self.cluster)
        else:
            target = ""
            device_fn = ""
        return (target, device_fn)

    def remove_training_directory(self, train_dir):
        try:
            logging.info(f"{str_task(self.task)}: "\
                         "Removing existing train directory.")
            gfile.DeleteRecursively(train_dir)
        except:
            logging.error(f"{str_task(self.task)}: Failed to delete directory "\
                          f"{train_dir} when starting a new model. "\
                          "Please delete it manually and try again.")

    def get_meta_filename(self, start_new_model, train_dir):
        if start_new_model:
            logging.info(f"{str_task(self.task)}: "\
                         "Flag 'start_new_model' is set. Building a new model.")
            return None
        latest_checkpoint = tf.train.latest_checkpoint(train_dir)
        if not latest_checkpoint:
            logging.info(f"{str_task(self.task)}: "\
                         "No checkpoint file found. Building a new model.")
            return None
        meta_filename = latest_checkpoint + ".meta"
        if not gfile.Exists(meta_filename):
            logging.info(f"{str_task(self.task)}: No meta graph file found. "\
                         "Building a new model.")
            return None
        else:
            return meta_filename

    def recover_model(self, meta_filename):
        logging.info(f"{str_task(self.task)}: "\
                     f"Restoring from meta graph file {meta_filename}")
        return tf.train.import_meta_graph(meta_filename)

    def build_model(self, model, reader):
        label_loss_fn = find_class_by_name(FLAGS.label_loss, [losses])()
        optimizer_class = find_class_by_name(FLAGS.optimizer, [tf.train])
        build_graph(reader=reader,
                    model=model,
                    optimizer_class=optimizer_class,
                    clip_gradient_norm=FLAGS.clip_gradient_norm,
                    train_data_pattern=FLAGS.train_data_pattern,
                    label_loss_fn=label_loss_fn,
                    base_learning_rate=FLAGS.base_lr,
                    learning_rate_decay=FLAGS.lr_decay,
                    learning_rate_decay_examples=FLAGS.lr_decay_examples,
                    regularization_penalty=FLAGS.regularization_penalty,
                    num_readers=FLAGS.num_readers,
                    batch_size=FLAGS.batch_size,
                    num_epochs=FLAGS.num_epochs)
        return tf.train.Saver(max_to_keep=0, keep_checkpoint_every_n_hours=5)

def get_reader():
    feature_names, feature_sizes = get_feature_names_and_sizes(
            FLAGS.feature_names, FLAGS.feature_sizes)
    if FLAGS.frame_features:
        reader = YT8MFrameFeatureReader(feature_names=feature_names,
                                        feature_sizes=feature_sizes)
    else:
        reader = YT8MAggregatedFeatureReader(feature_names=feature_names,
                                             feature_sizes=feature_sizes)
    return reader

class ParameterServer(object):
    def __init__(self, cluster, task):
        self.cluster = cluster
        self.task = task

    def run(self):
        logging.info(f"{str_task(self.task)}: Starting parameter server "\
                     f"within cluster {self.cluster.as_dict()}.")
        server = start_server(self.cluster, self.task)
        server.join()

def start_server(cluster, task):
    if not task.type:
        raise ValueError(f"{str_task(task)}: The task type must be specified.")
    if task.index is None:
        raise ValueError(f"{str_task(task)}: The task index must be specified.")
    return tf.train.Server(tf.train.ClusterSpec(cluster),
                           protocol="grpc",
                           job_name=task.type,
                           task_index=task.index)

def str_task(task):
    return f"/job:{task.type}/task:{task.index}"

def main(unused_argv):
    env = json.loads(os.environ.get("TF_CONFIG", "{}"))
    cluster_data = env.get("cluster", None)
    cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None
    task_data = env.get("task", None) or {"type": "master", "index": 0}
    task = type("TaskSpec", (object,), task_data)
    logging.set_verbosity(tf.logging.INFO)
    logging.info(f"{str_task(task)}: Tensorflow version: {tf.__version__}.")
    if not cluster or task.type == "master" or task.type == "worker":
        model = find_class_by_name(FLAGS.model,
                                   [frame_level_models, video_level_models])()
        reader = get_reader()
        model_exporter = ModelExporter(frame_features=FLAGS.frame_features,
                                       model=model,
                                       reader=reader)
        Trainer(cluster, task, FLAGS.train_dir, model, reader, model_exporter,
                FLAGS.log_device_placement, FLAGS.export_model_steps,
                FLAGS.max_steps).run(start_new_model=FLAGS.start_new_model)
    elif task.type == "ps":
        ParameterServer(cluster, task).run()
    else:
        raise ValueError(f"{str_task(task)}: Invalid task_type: {task.type}.")

if __name__ == "__main__":
    tf.app.run()
