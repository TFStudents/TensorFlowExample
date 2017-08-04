#user/bin
import tensorflow as tf

que=tf.FIFOQueue(3,[tf.int32])
node1= tf.constant([1,1])
que.enqueue_many(node1)
#que.enqueue_many([122])# 会报错
print(que.dequeue())
print(que.dequeue())

print(que.dequeue())