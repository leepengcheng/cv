#coding:utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
'''
å®¸èŒ¬ç…¡é�‚åœ­â–¼ Wx+b=y
ç¼�æ¬�åš­100æ¶“ç�‚æ�ˆæ’³å�†é�œå±½î‡®æ�´æ—‚æ®‘æ�ˆæ’³åš­y
å§¹ä¿‰é�œå®�é�¨å‹¬å«™é�šå �â‚¬ï¿½
é�—ç �->[0.100, 0.200] b->0.300é�¨å‹«äº¸ç»‚è�¤â–¼æ�´ï¿½
'''

# é�¢ç†¸å�šé—…å¿”æº€é��ç‰ˆåµ� é�¬è¯²å�¡ 100 æ¶“î�†å�£é�¨å‹®ç·­é��ãƒ¥â‚¬ç…�æ‹°æ�ˆæ’³åš­é�Šï¿½
x_data = np.float32(np.random.rand(100, 2)) # shape=(100,2)
y_data = np.dot(x_data,[[0.1],[0.2]]) + 0.300# shape=(1,2)*(2,100)=(1,100)


W = tf.Variable(tf.random_uniform([2, 1], -1.0, 1.0))
B = tf.Variable(tf.zeros([1]))
# é�‹å‹¯â‚¬çŠµåš�é�¬Ñ„Ä�é�¨ï¿½ x*W+b=y
y = tf.matmul(x_data,W) + B

# å¯¤è™¹ç�›é�¹ç†·ã�‘é�‘èŠ¥æšŸ->é�§å›¨æŸŸå®¸î†¼åš±é��ï¿½
loss = tf.reduce_mean(tf.square(y - y_data))
# loss=tf.reduce_mean(-y_data*tf.log(y))
# ç’�å‰§ç–†ç’�î… ç²Œæµ¼æ¨ºå¯²é�£ã„§æ®‘ç€›ï¸¿ç¯„é�œå›¦è´Ÿ0.5
train  = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
# é�’æ¿†î��é�–æ §å½‰é–²ï¿½
init = tf.global_variables_initializer()
# é�šîˆšå§©æµ¼æ°³ç˜½/æµœã‚„ç°°æµ¼æ°³ç˜½(é�¢ã„¤ç°¬Ipython)
sess = tf.Session()
# sess =tf.InteractiveSession()
sess.run(init)
# 201æ�©î…�å”¬
for step in range(100):
    l,_,w,b=sess.run([loss,train,W,B])
    print("step: %4d  loss:%s   w:%s b:%s\n"%(step,l,w,b))
sess.close()
E:\Anaconda\python.exe E:\Anaconda\cwp.py E:\Anaconda "E:/Anaconda/python.exe" "E:/Anaconda/Scripts/jupyter-notebook-script.py" %USERPROFILE%