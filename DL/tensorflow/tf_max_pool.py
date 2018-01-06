# import tensorflow as tf  
  
# a=tf.constant([  
#         1.0,2.0,3.0,4.0,  
#         5.0,6.0,7.0,8.0,  
#         8.0,7.0,6.0,5.0,  
#         4.0,3.0,2.0,1.0,  
#         4.0,3.0,2.0,1.0,  
#          8.0,7.0,6.0,5.0,  
#          1.0,2.0,3.0,4.0,  
#          5.0,6.0,7.0,8.0])  
  
# a=tf.reshape(a,[1,4,4,2])  
# maxPooling=tf.nn.max_pool(a,[1,2,2,1],[1,1,1,1],padding="VALID")
# with tf.Session() as sess:
#     print(sess.run(a))
#     print(sess.run(maxPooling))


import tensorflow as tf  
input = tf.Variable(tf.random_normal([1,3,3,5]))  
#核大小(1行-1列)-5通道-输出值1个
filter = tf.Variable(tf.random_normal([1,1,5,1])) 
op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')  
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    a=sess.run(input)
    b=sess.run(filter)
    c=sess.run(op)
    print(a)
    print(b)
    print(c)