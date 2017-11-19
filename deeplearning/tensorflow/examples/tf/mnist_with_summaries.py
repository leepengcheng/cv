# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A simple MNIST classifier which displays summaries in TensorBoard.

 This is an unimpressive MNIST model, but it is a good example of using
tf.name_scope to make a graph legible in the TensorBoard graph explorer, and of
naming summary tags so that they are grouped meaningfully in TensorBoard.

It demonstrates the functionality of every TensorBoard dashboard.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
from random import random
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None


def train():
  # 导入数据
  mnist = input_data.read_data_sets(FLAGS.data_dir,
                                    one_hot=True,
                                    fake_data=FLAGS.fake_data)
  #创建交互回话
  sess = tf.InteractiveSession()

  # 创建多层网络
  # 创建层：输入层-名称
  with tf.name_scope('Inputlayer'):
    x = tf.placeholder(tf.float32, [None, 784], name='InputImage')#输入图片
    y_ = tf.placeholder(tf.float32, [None, 10], name='InputImageLabel')#图片的标签

  #创建层:维度转换层
  with tf.name_scope('ReshapeImage'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('Image_Input', image_shaped_input, 10) #创建图片数据汇总->最大显示10个

  # We can't initialize these variables to 0 - the network will get stuck.
  def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('Summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('Mean', mean)
      with tf.name_scope('ScopeStddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('Stddev', stddev)
      tf.summary.scalar('Max', tf.reduce_max(var))
      tf.summary.scalar('Min', tf.reduce_min(var))
      tf.summary.histogram('Histogram_Var', var)

  def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.

    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
      # This Variable will hold the state of the weights for the layer
      with tf.name_scope('Weight'):
        weights = weight_variable([input_dim, output_dim])
        variable_summaries(weights)
      with tf.name_scope('Bias'):
        biases = bias_variable([output_dim])
        variable_summaries(biases)
      with tf.name_scope('WX_B'):
        preactivate = tf.matmul(input_tensor, weights) + biases
        tf.summary.histogram('Histogram_Preactivate', preactivate)
      activations = act(preactivate, name='AfterAction')
      tf.summary.histogram('Histogram_activations', activations)
      return activations

  hidden1 = nn_layer(x, 784, 500, 'Layer1')

  with tf.name_scope('Dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('DropProb', keep_prob)
    dropped = tf.nn.dropout(hidden1, keep_prob)

  # Do not apply softmax activation yet, see below.
  y = nn_layer(dropped, 500, 10, 'Layer2', act=tf.identity)

  with tf.name_scope('Scope_Cross-entropy'):
    # The raw formulation of cross-entropy,
    #
    # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
    #                               reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the
    # raw outputs of the nn_layer above, and then average across
    # the batch.
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    with tf.name_scope('Scope_Cross-entropy-reducemean'):
      cross_entropy = tf.reduce_mean(diff)
  tf.summary.scalar('Loss_CrossEntropy', cross_entropy)

  with tf.name_scope('Scope_Train'):
    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
        cross_entropy)

  with tf.name_scope('Accuracy'):
    with tf.name_scope('AccuracyArgmax'):
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('AccuracyPercent'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('Scalar_Accuracy', accuracy)

  # 合并图中统计数据,返回值为string 类型的tensor
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)#训练输出 流
  test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test') #测试输出流
  tf.global_variables_initializer().run()

  # Train the model, and also write summaries.
  # Every 10th step, measure test-set accuracy, and write test summaries
  # All other steps, run train_step on training data, & add training summaries

  def feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train or FLAGS.fake_data:
      xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
      k = FLAGS.dropout
      # k=random()
    else:
      xs, ys = mnist.test.images, mnist.test.labels
      k = 1.0
    return {x: xs, y_: ys, keep_prob: k}

  for i in range(FLAGS.max_steps):
    if i % 10 == 0:  # 每隔10步刷新统计和准确率
      summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
      test_writer.add_summary(summary, i) #写出第N步的统计信息
      print('Accuracy at step %s: %s' % (i, acc))#输出准确率
    else:  # Record train set summaries, and train
      if i % 100 == 99:  # 每100步刷新状态信息
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merged, train_step],
                              feed_dict=feed_dict(True),
                              options=run_options,
                              run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        train_writer.add_summary(summary, i) #写出第N步的统计信息
        print('Adding run metadata for', i)
      else:  # Record a summary
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
        train_writer.add_summary(summary, i) #写出第N步的统计信息
  train_writer.close()
  test_writer.close()


def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  train()


if __name__ == '__main__':
  from os.path import dirname,join
  path=dirname(dirname(dirname(__file__)))
  parser = argparse.ArgumentParser()
  parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                      default=False,
                      help='If true, uses fake data for unit testing.')
  parser.add_argument('--max_steps', type=int, default=1000,
                      help='Number of steps to run trainer.')
  parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
  parser.add_argument('--dropout', type=float, default=0.6,
                      help='Keep probability for training dropout.')
  parser.add_argument('--data_dir', type=str, default=join(path,"data\mnist"),
                      help='Directory for storing input data')
  parser.add_argument('--log_dir', type=str, default=join(path,"data\summary"),
                      help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
