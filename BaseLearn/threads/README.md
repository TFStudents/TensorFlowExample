# 队列和线程
## 队列概述
- 多个线程准备实例并放在队列中
- 训练线程 从队列中取出训练操作op
- Session 是多线程的，所以多线程可以使用同一个Session
- 所有的线程都可以一起被关闭 并且当队列停止时要合理关闭

设计两个类  一起使用
## tf.train.Coordinator  
> 协作者   
> 帮助多线程的合理停止和异常处理   
- 实例
```python
import tensorflow as tf
import threading
import time

# 创建一个方法 通过coord来控制线程
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

# Create 3 threads that run 'MyLoop()'
threads = [threading.Thread(target=MyLoop, args=(coord,)) for i in range(3)] 
for t in threads:
  t.start() # start a thread
coord.join(threads) # wait util thread stop
```

### tf.train.QueueRunner
> 队列的执行者    
> 创建一系列线程，在同一个队列中组织tensors的排队执行 
- 实例 
```python
#1 create Runner
b =tf.constant(2,dtype=tf.float16,shape=[1])
que=tf.RandomShuffleQueue(3,1,dtypes=[tf.float16],shapes=())#shapes一定要写 否则会报错 
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
```
![流程图](https://github.com/TFStudents/Tensorflow/blob/master/Resource/thread_queque.png)

