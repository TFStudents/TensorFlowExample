#user/bin
import tensorflow as tf
#---常数----
node1= tf.constant([True,True])
node2= tf.constant(12)
node3=tf.constant("hello")
node4=tf.constant("world")
print(node1)#打印Tensor("Const:0", shape=(2,), dtype=bool)
print(node2)#打印Tensor("Const_1:0", shape=(), dtype=int32)
print(node3)#打印Tensor("Const_2:0", shape=(), dtype=string)
#---会话---
sess=tf.Session()#获取会话
print(sess.run(node1))#打印[True,True]
print(sess.run(node2))#打印12
print(sess.run(node3))#打印b'hello'
print(sess.run(node4))#打印b'world'
#---加法----
node5=tf.constant(-1)
n_add=tf.add(node2,node5)
print(sess.run(n_add))#打印11
node6=tf.constant(-2)
n_add2=node5+node6
print(sess.run(n_add2))#打印-3 说明两种形式的加法也可以的 
#---占位符---
p1=tf.placeholder(tf.float32)
p2=tf.placeholder(tf.float32)
n_add3=p1+p2
a=sess.run(n_add3,{p1:23,p2:32})#用法正确 赋值以json的格式 进行赋值
#a=(n_add3,{p1:23,p2:32})错误 只能和sess一起使用
print(a)#55.0
print(sess.run(n_add3,{p1:23,p2:32}))#用法正确 打印55.0
#---变量---
v1=tf.Variable(12)
v2=tf.Variable(23)#定义一个变量
init = tf.global_variables_initializer()#进去全部变量的初始化
sess.run(init)
x=tf.placeholder(tf.int32)
a_line=v1*x+v2#使用和常量类似
v3=sess.run(a_line,{x:[1,2]})
print(v3)#[35 47]
#重新赋值
v1=tf.assign(v1,-2)
v2=tf.assign(v2,3)
add4=v1+v2
v4=sess.run(add4)
print(v4)#1
