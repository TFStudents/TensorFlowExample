# 共享变量
[share_variable.py](./share_variable.py)
## 存在问题
- 代码块 A
```
sess=tf.Session()
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
```
当我们运行代码块A时,方法内会产生4个变量，当运算大时，会产生过多的内存消耗，那怎么解决这个问题呢？  
有如下方案   
- 代码块B
```
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
```
当我们执行两遍printVariable方法时，确实解决了这个问题，只有两个变量，单存在一些问题
1. 当创建变量时，必须指定一些如类型，形状等参数
2. 当代码改变时，可能创建的变量会更多

tensorflow提供了一种更方便的方式提供相应的功能，直接看代码
- 代码块C
```
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
'''
打印的信息
v1.name=boo/v1:0
v2.name=boo/v2:0
----------boo--1-----------
v1.name=boo/v1:0
v2.name=boo/v2:0
----------boo--2-----------
v1.name=boo/v1:0
v2.name=boo/v2:0
----------boo--3-----------
v1.name=boo/v1:0
v2.name=boo/v2:0
----------boo--4-----------
v1.name=boo/v1:0
v2.name=boo/v2:0

''' 
```
上面虽然调用了好几次，但是变量是共享的，所以只产生了2个变量,而tf就是这么进行变量共享的，先看两个关键函数
## tf.get_variable
[tf.get_variable](https://www.tensorflow.org/api_docs/python/tf/get_variable)
- 参数列表
```
get_variable(
    name,
    shape=None,
    dtype=None,
    initializer=None,
    regularizer=None,
    trainable=True,
    collections=None,
    caching_device=None,
    partitioner=None,
    validate_shape=True,
    use_resource=None,
    custom_getter=None
)
```
传入值时，除name外shape也是必须传的
```
v=tf.get_variable("v1",shape=[1]，initializer=tf.constant_initializer(1)) #这样一个变量就创建了
```
- 它是怎么工作的？
1. 名字： 传入相应参数后，v将会被创建，它的名字就是scope_name+variable_name  
在代码块C中,scope_name="boo",variable_name="v1"或"v2" 则v1的全名就是"boo/v1:0" 参考打印信息
2. 重用：要确保tf.get_variable_scope().reuse == True ,调用就scope.reuse_variables()就可以了 但不能直接设置为false

## tf.variable_scope
[tf.variable_scope](https://www.tensorflow.org/api_docs/python/tf/variable_scope)
- 参数列表 
```
variable_scope(
    name_or_scope,
    default_name=None,
    values=None,
    initializer=None,
    regularizer=None,
    caching_device=None,
    partitioner=None,
    custom_getter=None,
    reuse=None,
    dtype=None,
    use_resource=None
)
```
一般只传name值就可以了，作用就是为tf.get_variable做名字区分以及重用控制，当然还可以控制initializer   
差不多这些内容就够用了 ，具体细节点击链接看看[本文参考地址](https://www.tensorflow.org/programmers_guide/variable_scope)

## end 

