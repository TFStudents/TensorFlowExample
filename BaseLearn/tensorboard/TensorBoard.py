#python
import tensorflow as tf
node1=tf.constant(2,tf.float32)#定义节点，值为2
node2=tf.constant(3,tf.float32)#定义节点，值为3
node=tf.add(node1,node2)
print(node1)
with tf.Session() as sess:
    print(sess.run(node))
    tf.global_variables_initializer().run()
    writer = tf.summary.FileWriter("test_1", sess.graph)
