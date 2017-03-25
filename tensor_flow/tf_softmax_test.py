import tensorflow as tf
from  tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../data/mnist", one_hot=True)

learn_rate=0.1
epochs=2
batch_size=100

#输入数据
x=tf.placeholder(tf.float32,shape=[None,784])
y=tf.placeholder(tf.float32,shape=[None,10])


#待计算数据
W=tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#模型
model=tf.nn.softmax(tf.matmul(x,W)+b)

#损失函数-交叉熵
cost=tf.reduce_mean(-tf.reduce_sum(y*tf.log(model),1))
cost1=tf.reduce_mean(-y*tf.log(model))

#训练目标
tran_step=tf.train.GradientDescentOptimizer(0.1).minimize(cost)

#初始化变量
init=tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    total_batch=mnist.train.num_examples//batch_size
    for i in range(epochs):
        for j in range(total_batch):
            x_t,y_t=mnist.train.next_batch(100)
            sess.run(tran_step,feed_dict={x:x_t,y:y_t})
            val_cost,val_cost1,_=sess.run([cost,cost1,tran_step],feed_dict={x:x_t,y:y_t})
        print("epoch:%s---cost:%s-%s"%(i,val_cost,val_cost1))
        correct_prediction=tf.equal(tf.argmax(model,1),tf.argmax(y,1))
    accury=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    print(accury.eval({x:mnist.test.images,y:mnist.test.labels}))