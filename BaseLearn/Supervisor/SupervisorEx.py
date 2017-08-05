# user/bin
import tensorflow as tf
import threading
'''
this is test for how to use the Supervisor 
'''
def runSum():
    a=tf.constant(12,dtype=tf.int8)
    b=tf.constant(10,dtype=tf.int8)
    sv=tf.train.Supervisor(logdir="./test1")
    with sv.managed_session() as sess:
        for i in range(10):
            if sv.should_stop():
                return
            print(sess.run([a,b]))

runSum()
