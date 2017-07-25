#user/bin
import tensorflow as tf
sess=tf.Session()
print("----------0-------------")
def printVariable():
	v1=tf.Variable(1,name="v1")
	v2=tf.Variable(1,name="v2")
	init=tf.global_variables_initializer()#变量的初始化 初始化当前module所有的变量
	sess.run(init)
	return v1+v2
	
a1=printVariable()#两个变量
b1=printVariable()#两个变量 总4个变量
print(sess.run(a1))
print(sess.run(b1))
print("----------1-------------")
v_dict={
	"v1":tf.Variable(1,name="v1"),
	"v2":tf.Variable(1,name="v2")
}
def printVariable(dicts):
	v1=dicts["v1"]
	v2=dicts["v2"]
	return v1.initialized_value()+v2.initialized_value()
a2=printVariable(v_dict)#两个变量
b2=printVariable(v_dict)#两个变量 总2两个
print(sess.run(a2))
print(sess.run(b2))

print("----------2-------------")

def printVariable2():
	v1=tf.get_variable("v1",[1],initializer=tf.constant_initializer(1))
	print("v1.name="+v1.name)
	v2=tf.get_variable("v2",[1],initializer=tf.constant_initializer(2))
	print("v2.name="+v2.name)
	return v1+v2
with tf.variable_scope("boo") as boo:
	printVariable2()
	print("----------boo--1-----------")
	boo.reuse_variables()
	printVariable2()
	print("----------boo--2-----------")
	printVariable2()
	print("----------boo--3-----------")
	printVariable2()
	print("----------boo--4-----------")
	printVariable2()

#------------variable_scope--------------------------
def onTestScope():
	with tf.variable_scope("boo") as boo:
		v=tf.get_variable("v_1",[20],initializer=tf.constant_initializer(2))
		print("v.name="+v.name)
		print("boo.name="+boo.name)
	with tf.variable_scope(boo):
		v2=tf.get_variable("v_2",[20],initializer=tf.constant_initializer(3))
		v1=v+v2
		print("v2.name="+v2.name)
		print("v1.name="+v1.name)
		return v1
with tf.variable_scope("test") as sc:
	onTestScope()
	sc.reuse_variables()
	onTestScope()

# with tf.variable_scope("ABC",initializer=tf.constant_initializer(233)):
# 	v=tf.get_variable("v3",[1])
# 	with tf.Session() as sess:
# 		print(sess.run(v.variable_value()))
	


