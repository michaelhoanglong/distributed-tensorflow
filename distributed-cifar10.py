import argparse
import sys
import traceback
import os
import errno
import stat
import logging
import zipfile
import time
import trainingalgorithm

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from tensorflow.contrib.session_bundle import exporter
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import (
    signature_constants, signature_def_utils, tag_constants, utils)
from tensorflow.python.util import compat

FLAGS = None

def get_session(sess):
  session = sess
  while type(session).__name__ != 'Session':
    session = session._sess
  return session

def zip(src, dstZipFile):
    zf = zipfile.ZipFile(dstZipFile, "w", zipfile.ZIP_DEFLATED)
    abs_src = os.path.abspath(src)
    for dirname, subdirs, files in os.walk(src):
        for filename in files:
            absname = os.path.abspath(os.path.join(dirname, filename))
            arcname = absname[len(abs_src) + 1:]
            print ('zipping %s as %s' % (os.path.join(dirname, filename),arcname))
            zf.write(absname, arcname)
    zf.close()

def saved_model(sess, prediction_signature, legacy_init_op):
  print("Export the saved model to {}".format(FLAGS.model_dir))

  sess.graph._unsafe_unfinalize()

  export_path_base = FLAGS.model_dir
  export_path = os.path.join(
      compat.as_bytes(export_path_base),
      # TODO: change this to model version later 
      compat.as_bytes('1'))
  # export_path = os.path.join(
  #   tf.compat.as_bytes(export_path_base),
  #   tf.compat.as_bytes('1'))
  print('Exporting trained model to', export_path)

  try:
    builder = saved_model_builder.SavedModelBuilder(export_path)
    # builder = tf.saved_model.builder.SavedModelBuilder(export_path)


    builder.add_meta_graph_and_variables(
        sess,
        [tag_constants.SERVING],
        clear_devices=True,
        signature_def_map={
            # signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            # prediction_signature,
            'predict_images':
              prediction_signature,
        },
        #legacy_init_op=legacy_init_op)
        legacy_init_op=legacy_init_op)

    # builder.add_meta_graph_and_variables(
    #   sess, [tf.saved_model.tag_constants.SERVING],
    #   signature_def_map={
    #       'predict_images':
    #           prediction_signature,
    #   },
    #   legacy_init_op=legacy_init_op)

    sess.graph.finalize()

    builder.save()
    print('Done exporting!')

    # sess.graph.finalize()

    # builder.save()
  except Exception as e:
    print("Fail to export saved model, exception: {}".format(e))

def main(_):
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")

  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":
    #print to output file
    orig_stdout = sys.stdout
    # os.makedirs(FLAGS.log_dir, exist_ok=True)
    f = open(FLAGS.log_dir + '/output.txt' , 'w+')
    sys.stdout = f
    start_time = time.time()

    try:
      # Assigns ops to the local worker by default.
      with tf.device(tf.train.replica_device_setter(
          worker_device="/job:worker/task:%d" % FLAGS.task_index,
          cluster=cluster)):

        # Import data
        #dataset = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
        # PARAMS
        _IMAGE_SIZE = 32
        _IMAGE_CHANNELS = 3
        _NUM_CLASSES = 10
        from data import get_data_set
        train_x, train_y = get_data_set("train")
        test_x, test_y = get_data_set("test")
       
        
        # Build Deep MNIST model...
        # keys_placeholder = tf.placeholder(tf.int32, shape=[None, 1])
        # keys = tf.identity(keys_placeholder)

        # TODO: Change this 3 lines for new model implementation
        # x = trainingalgorithm.getDataTensorPlaceHolder()
        x = tf.placeholder(tf.float32, shape=[None, _IMAGE_SIZE * _IMAGE_SIZE * _IMAGE_CHANNELS])
        x = tf.identity(x, name='x')
        #serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
        #feature_configs = {'x': tf.FixedLenFeature(shape=[784], dtype=tf.float32),}
        #feature_configs = {'x': tf.FixedLenFeature(shape=[None, img_size, img_size, num_channels], dtype=tf.float32),}
        #tf_example = tf.parse_example(serialized_tf_example, feature_configs)
        #x = tf.identity(tf_example['x'], name='x')  # use tf.identity() to assign name
        #y_ = trainingalgorithm.getLabelTensorPlaceHolder()
        y_ = tf.placeholder(tf.float32, [None, _NUM_CLASSES])
        #y_conv, keep_prob = trainingalgorithm.trainingAlgorithm(x)
        from model import model, lr
        x, y, y_conv, y_pred_cls, global_step_notuse, learning_rate_notuse = model(x,y_)
        y_conv = tf.identity(y_conv, name='y_conv')
        #keep_prob = tf.identity(keep_prob, name='keep_prob')
        
        


        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

        global_step = tf.contrib.framework.get_or_create_global_step()

        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy, global_step=global_step)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # prediction_signature = signature_def_utils.build_signature_def(
        #       inputs={
        #           "keys": utils.build_tensor_info(keys_placeholder),
        #           "features": utils.build_tensor_info(x)
        #       },
        #       outputs={
        #           "keys": utils.build_tensor_info(keys),
        #           "prediction": utils.build_tensor_info(correct_prediction)
        #       },
        #       method_name=signature_constants.PREDICT_METHOD_NAME)

        tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
        tensor_info_y = tf.saved_model.utils.build_tensor_info(y_conv)
        #tensor_info_keepprob = tf.saved_model.utils.build_tensor_info(keep_prob)

        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'images': tensor_info_x},
                outputs={'scores': tensor_info_y},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

        legacy_init_op = tf.group(
            tf.initialize_all_tables(), name="legacy_init_op")
        # legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')


      # The StopAtStepHook handles stopping after running given steps.
      hooks=[tf.train.StopAtStepHook(last_step=1000)]

      # The MonitoredTrainingSession takes care of session initialization,
      # restoring from a checkpoint, saving to a checkpoint, and closing when done
      # or an error occurs.
      # Worker with task_index = 0 is the Master Worker.
      # checkpoint_dir=FLAGS.log_dir,
      saver = tf.train.Saver()
      with tf.train.MonitoredTrainingSession(master=server.target,
                                             is_chief=(FLAGS.task_index == 0),
                                             hooks=hooks) as mon_sess:
        i = 0
        cur_batch = 0
        while not mon_sess.should_stop():
          # Run a training step asynchronously.
          #batch = dataset.train.next_batch(50)
          if(cur_batch >= len(train_x)):
            cur_batch = 0
          batch_x = train_x[cur_batch : cur_batch+50]
          batch_y = train_y[cur_batch : cur_batch+50]
          cur_batch = cur_batch+50
          if i % 100 == 0:
            # train_accuracy = mon_sess.run(accuracy, feed_dict={
            #     x: batch[0], y_: batch[1], keep_prob: 0.9})
            train_accuracy = mon_sess.run(accuracy, feed_dict={
                x: batch_x, y_: batch_y})
            print('Global_step %s, task:%d_step %d, training accuracy %g' % (tf.train.global_step(mon_sess, global_step), FLAGS.task_index, i, train_accuracy))
          #mon_sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
          mon_sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})
          i = i + 1
        stop_time = time.time()
        print('Training completed!')
        print('Number of parameter servers: %s' % len(ps_hosts))
        print('Number of workers: %s' % len(worker_hosts))
        print('Excution time: %s' % (stop_time - start_time))
        if FLAGS.task_index == 0:
          saved_model(get_session(mon_sess), prediction_signature, legacy_init_op)
    except Exception as e:
      print(traceback.format_exc())
    finally:
        sys.stdout = orig_stdout
        f.close()

    os.system("cat " + FLAGS.log_dir + "/output.txt")
    if(FLAGS.task_index == 0):
      try:
        os.makedirs(FLAGS.model_dir)
      except OSError as e:
        if e.errno != errno.EEXIST:
          raise
      os.system("cp " + FLAGS.log_dir + "/distributed-tensorflow/trainingalgorithm.py" + " " + FLAGS.model_dir)
      os.system("cp " + FLAGS.log_dir + "/output.txt" + " " + FLAGS.model_dir)
      zipFileName = FLAGS.model_dir + FLAGS.zip_name
      zip(FLAGS.model_dir,zipFileName)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  # Flags for defining the tf.train.ClusterSpec
  parser.add_argument(
      "--ps_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--worker_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--job_name",
      type=str,
      default="",
      help="One of 'ps', 'worker'"
  )
  # Flags for defining the tf.train.Server
  parser.add_argument(
      "--task_index",
      type=int,
      default=0,
      help="Index of task within the job"
  )
  # Flags for specifying input/output directories
  parser.add_argument(
      "--data_dir",
      type=str,
      default="/tmp/mnist_data",
      help="Directory for storing input data")
  parser.add_argument(
      "--model_dir",
      type=str,
      default="/home/ubuntu/s3-drive/model_dir",
     help="Directory for output model (must be a shared Directory)")
  parser.add_argument(
      "--zip_name",
      type=str,
      default="model.zip")
  parser.add_argument(
      "--log_dir",
      type=str,
      default="/home/ubuntu",
     help="Directory for output log")
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)