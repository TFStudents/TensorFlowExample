# 训练助手
> 当需要做以日计的训练数据 有如下需求
- 处理关机和异常的清除
- 可以从关机或异常中重新运行 
- 可以TensorBoard中被监控

关键类：[tf.train.Supervisor](https://www.tensorflow.org/api_docs/python/tf/train/Supervisor)

```python
def runSum():
    a=tf.constant(12,dtype=tf.int8)
    b=tf.constant(10,dtype=tf.int8)
    sv=tf.train.Supervisor(logdir="./test1")
    with sv.managed_session() as sess:
        for i in range(10):
            if sv.should_stop():
                return
            print(sess.run([a,b]))
'''
[12, 10]
[12, 10]
[12, 10]
[12, 10]
[12, 10]
[12, 10]
[12, 10]
[12, 10]
[12, 10]
[12, 10]
'''
```
[实例代码](./SupervisorEx.py)
- 未完待续
