import argparse
import sys
import os
import time
import stat
import logging

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from tensorflow.contrib.session_bundle import exporter
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import (
    signature_constants, signature_def_utils, tag_constants, utils)
from tensorflow.python.util import compat

FLAGS = None

def deepnn(x):
  x_image = tf.reshape(x, [-1, 28, 28, 1])

  W_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  h_pool1 = max_pool_2x2(h_conv1)

  W_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  h_pool2 = max_pool_2x2(h_conv2)

  W_fc1 = weight_variable([7 * 7 * 64, 1024])
  b_fc1 = bias_variable([1024])

  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  W_fc2 = weight_variable([1024, 10])
  b_fc2 = bias_variable([10])

  y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def write_log(content):
  filename = FLAGS.log_dir + '/' + 'training_logs_task_' + str(FLAGS.task_index) + '.txt'
  os.makedirs(os.path.dirname(FLAGS.log_dir), exist_ok=True)
  os.chmod(FLAGS.log_dir, stat.S_IRWXO | stat.S_IRWXG | stat.S_IRWXU)
  with open(filename,'a') as file:
    file.write(content)

def get_session(sess):
  session = sess
  while type(session).__name__ != 'Session':
    session = session._sess
  return session

def saved_model(sess, model_signature, legacy_init_op):
  logging.info("Export the saved model to {}".format(FLAGS.log_dir))

  sess.graph._unsafe_unfinalize()

  export_path_base = FLAGS.log_dir
  export_path = os.path.join(
      compat.as_bytes(export_path_base),
      # TODO: change this to model version later 
      compat.as_bytes(str(1)))

  try:
    builder = saved_model_builder.SavedModelBuilder(export_path)
    builder.add_meta_graph_and_variables(
        sess,
        [tag_constants.SERVING],
        clear_devices=True,
        signature_def_map={
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            model_signature,
        },
        #legacy_init_op=legacy_init_op)
        legacy_init_op=legacy_init_op)

    sess.graph.finalize()

    builder.save()
  except Exception as e:
    logging.error("Fail to export saved model, exception: {}".format(e))

def main(_):
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")

  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

  # config
  batch_size = 100
  learning_rate = 0.001
  training_epochs = 20
  #print to output file
  orig_stdout = sys.stdout
  f = open(FLAGS.log_dir + '/output.txt' , 'w+')
  sys.stdout = f

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":

    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):

      # Import data
      mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

      # Build Deep MNIST model...
      keys_placeholder = tf.placeholder(tf.int32, shape=[None, 1])
      keys = tf.identity(keys_placeholder)
      x = tf.placeholder(tf.float32, [None, 784])
      y_ = tf.placeholder(tf.float32, [None, 10])
      y_conv, keep_prob = deepnn(x)

      cross_entropy = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

      global_step = tf.contrib.framework.get_or_create_global_step()

      train_op = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy, global_step=global_step)
      correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

      # Create summary for cost and accuracy
      tf.summary.scalar("cost",cross_entropy)
      tf.summary.scalar("accuracy", accuracy)

      # Merge all summaries into a single "operation" to be executed in one session
      summary_op = tf.summary.merge_all()
      init_op = tf.initialize_all_variables()
      saver = tf.train.Saver()
      print("Variables initialized...")

    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                            global_step=global_step,
                            init_op=init_op)

    begin_time = time.time()
    frequency = 100
    with sv.managed_session(server.target) as sess:
      '''
      # is chief
      if FLAGS.task_index == 0:
        sv.start_queue_runners(sess, [chief_queue_runner])
        sess.run(init_token_op)
      '''
      # create log writer object (this will log on every machine)
      writer = tf.summary.FileWriter(FLAGS.log_dir, graph=tf.get_default_graph())
          
      # perform training cycles
      start_time = time.time()
      for epoch in range(training_epochs):

        # number of batches in one epoch
        batch_count = int(mnist.train.num_examples/batch_size)

        count = 0
        for i in range(batch_count):
          batch_x, batch_y = mnist.train.next_batch(batch_size)
          
          # perform the operations we defined earlier on batch
          _, cost, summary, step = sess.run(
                          [train_op, cross_entropy, summary_op, global_step], 
                          feed_dict={x: batch_x, y_: batch_y})
          writer.add_summary(summary, step)

          count += 1
          if count % frequency == 0 or i+1 == batch_count:
            elapsed_time = time.time() - start_time
            start_time = time.time()
            print("Step: %d," % (step+1), 
                  " Epoch: %2d," % (epoch+1), 
                  " Batch: %3d of %3d," % (i+1, batch_count), 
                  " Cost: %.4f," % cost, 
                  " AvgTime: %3.2fms" % float(elapsed_time*1000/frequency))
            count = 0
        #os.system("s3cmd put /home/ubuntu/output.txt s3://fyp2017/output/<worker_id>/output.txt")   
      save_path = saver.save(sess, "/home/ubuntu/model/model.ckpt", global_step=global_step)
      print("Model saved in file: %s" % save_path)

      print("Test-Accuracy: %2.2f" % sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
      print("Total Time: %3.2fs" % float(time.time() - begin_time))
      print("Final Cost: %.4f" % cost)

    sv.stop()
    print("done")
  
  sys.stdout = orig_stdout
  f.close()
  #os.system("s3cmd put /home/ubuntu/output.txt s3://fyp2017/output/<worker_id>/output.txt")   
  os.system("cat " + FLAGS.log_dir + "/output.txt")

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
      "--log_dir",
      type=str,
      default="/home/ubuntu/log_dir",
     help="Directory for train logs")
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
