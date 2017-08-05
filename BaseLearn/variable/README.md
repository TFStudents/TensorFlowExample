# 变量
[源文件](./variable.py)   
地址[tf.Variable](https://www.tensorflow.org/api_docs/python/tf/Variable) 
```Python
#文件头先引入
import tensorflow as tf
```

## 如何创建？ 
- 构造函数
```Python
__init__(
    initial_value=None,
    trainable=True,
    collections=None,
    validate_shape=True,
    caching_device=None,
    name=None,
    variable_def=None,
    dtype=None,
    expected_shape=None,
    import_scope=None
)
```
从构造函数中可以看出，可以不传入任何参数，但是必须传入initial_value的值 不然会报错  
其他值可以任意组合传入 如下
```Python
v1=tf.Variable(0) 
v2=tf.Variable("hello world")
```
- 机器配置
```Python
# 将变量设置到cpu 0
with tf.device("/cpu:0"):
  v = tf.Variable(...)

#将变量设置到gpu 0
with tf.device("/gpu:0"):
  v = tf.Variable(...)
```

## 初始化
创建变量之后，需调用tf.global_variables_initializer()进行初始化  
才能进行调用 不然会报错的
```Python
#初始化方式1
init=tf.global_variables_initializer()#变量的初始化 初始化当前module所有的变量
sess=tf.Session()
sess.run(init)
print(v1)#Tensor("Variable/read:0", shape=(), dtype=int32)
#初始化方式2
v2=tf.Variable(2)
print(sess.run(v2.initialized_value()))#由于global_variables_initializer 其实不写也可以的
print(v2)# Tensor("Variable_1/read:0", shape=(), dtype=int32)
```
## 保存及读取变量
```Python
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
```

