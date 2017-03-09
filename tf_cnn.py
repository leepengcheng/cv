import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(r'D:\MyRepo\myml\data\mnist', one_hot=True)
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])


def weight_variable(shape):
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init)


def bias_variable(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)


#卷积
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


#最大池/采样层
def max_pool_2x2(x):
    return tf.nn.max_pool(
        x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


#输入N*(28-28)
x_image = tf.reshape(x, [-1, 28, 28, 1])
#第一层卷积32*(28-28)
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
#最大池化32*(14-14)
h_pool1 = max_pool_2x2(h_conv1)

#第二层卷积32*(14-14)->64*(14-14)
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
#64*(14-14)->64*(7-7)
h_pool2 = max_pool_2x2(h_conv2)

#全连接层
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])


#转换为列向量(batch_size,7*7*64)
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
# relu激活函数->(batch_size,1024)
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#dropout 层
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#输出层-1*1024 *1024-10->-1*10
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
#激活函数
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#损失函数-交叉熵
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
#优化函数
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# 验证测试数据
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
#转换为正确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
with tf.Session() as sess:
    #初始化函数
    sess.run(tf.global_variables_initializer())
    #训练20000轮epoch
    for i in range(20000):
        #每批次50张图片
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            #测试时keep_prob=1
            train_accuracy = accuracy.eval(
                feed_dict={x: batch[0],
                        y_: batch[1],
                        keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
            #训练时keep_prob=0.5
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            print("test accuracy %g" % accuracy.eval(feed_dict={
                x: mnist.test.images,
                y_: mnist.test.labels,
                keep_prob: 1.0
            }))
