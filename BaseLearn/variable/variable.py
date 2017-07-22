#关于variable
import tensorflow as tf

v1=tf.Variable(0,name="v1")#变量的创建

# 将变量设置到cpu 0
#with tf.device("/cpu:0"):
# v = tf.Variable(...)
#将变量设置到gpu 0
#with tf.device("/gpu:0"):
#  v = tf.Variable(...)
#初始化方式1
init=tf.global_variables_initializer()#变量的初始化 初始化当前module所有的变量
sess=tf.Session()
sess.run(init)
print(v1)#Tensor("Variable/read:0", shape=(), dtype=int32)
#初始化方式2
v2=tf.Variable(2)
print(sess.run(v2.initialized_value()))#由于global_variables_initializer 其实不写也可以的
print(v2)# Tensor("Variable_1/read:0", shape=(), dtype=int32)
# 保存session
saver=tf.train.Saver()
saver.save(sess,"./save")
v1=v1.assign(12)
print(sess.run(v1))#重新赋值后打印12

a=saver.restore(sess,"./save")# 读取模型
print(sess.run("v1:0"))#读取已经保存的变量v1 打印为0
# saver=tf.train.import_meta_graph("save.meta")
# saver.restore(sess,tf.train.latest_checkpoint("./"))
# print(sess.run("v1:0"))# 可以从内存中读取变量v1 打印为0