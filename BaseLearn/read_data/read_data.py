# user/bin

import tensorflow as tf
import threading
import time
""" 
    n2.csv content
    101,102,103,104,105
    201,203,204,202,205
    
    n1.csv content
    1,2,3,4,5
"""
# 1 read file
filename_queue = tf.train.string_input_producer(["n1.csv", "n2.csv"])
sess=tf.Session()
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
print("key=",key)
print("value=",value)
# Default values, in case of empty columns. Also specifies the type of the
# decoded result

# 2 decode file
record_defaults = [[1], [1], [1], [1], [1]]
col1, col2, col3, col4, col5 = tf.decode_csv(
    value, record_defaults=record_defaults)
print("c1-5=",col1," ",col2," ",col3," ",col4," ",col5)
features = tf.stack([col1, col2, col3, col4])
print("f=",features)
# 3 get the file content
with tf.Session() as sess:
  # Start populating the filename queue.
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)
  for i in range(5):
    # Retrieve a single instance:
    example, label = sess.run([features, col5])
    print(example)
    print(label)
  coord.request_stop()
  coord.join(threads)