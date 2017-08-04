# python/user
import tensorflow as tf
import threading
import time

#---------------queque sample use------------------------
que=tf.FIFOQueue(3,[tf.int32])
node1= tf.constant([1,1])
que.enqueue_many(node1)
#que.enqueue_many([122])# use the will make error
print(que.dequeue())
print(que.dequeue())
print(que.dequeue())


#-----------------Coordinator sample use--------------------
def MyLoop(coord):
  count=0
  while not coord.should_stop():
    print("count=",count)
    count=count+1
    time.sleep(1)
    if count==10:
      print("stop")
      coord.request_stop()

# Main thread: create a coordinator.
coord = tf.train.Coordinator()

# Create 10 threads that run 'MyLoop()'
threads = [threading.Thread(target=MyLoop, args=(coord,)) for i in range(3)]
for t in threads:
  t.start() # start a thread
coord.join(threads) # wait util thread stop

#-----------------QueueRunner example------------------------------------
#1 create Runner
b =tf.constant(2,dtype=tf.float16,shape=[1])
que=tf.RandomShuffleQueue(3,1,dtypes=[tf.float16],shapes=())
enqueue_op =que.enqueue_many(b)

qr = tf.train.QueueRunner(que,[enqueue_op])
#2 create Threads
sess=tf.Session()
coord = tf.train.Coordinator()
enqueue_threads = qr.create_threads(sess, coord=coord, start=True)
#3 use coord and queque
inputs=que.dequeue_many(10)+b
for step in  range(10):
    if coord.should_stop():
        break
    print("run")
    print(sess.run(inputs))
print("this request")
coord.request_stop()
coord.join(enqueue_threads)# wait thread to terminate



