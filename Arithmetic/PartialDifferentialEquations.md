# Partial Differential Equations（偏微分方程）

TensorFlow不仅仅用于机器学习。在这部分，给出一个使用TensorFlow仿真偏微分方程的例子，模拟一些雨点落在一个方形池塘的水面上。

目录

* 建立基础
* 便于计算的函数
* 定义概率密度函数
* 运行仿真

## 建立基础

导入一些库
```
# 导入用于仿真的库
import tensorflow as tf
import numpy as np

# 导入用于可视化的库
import PIL.Image
from io import BytesIO
from IPython.display import clear_output, Image, display

# 用于显示池塘水面状态的函数，水面状态以图像表示
def DisplayArray(a, fmt='jpeg', rng=[0,1]):
  """Display an array as a picture."""
  a = (a - rng[0])/float(rng[1] - rng[0])*255
  a = np.uint8(np.clip(a, 0, 255))
  f = BytesIO()
  PIL.Image.fromarray(a).save(f, fmt)
  clear_output(wait = True)
  display(Image(data=f.getvalue()))
  
  # 打开一个交互式的session
  sess = tf.InteractiveSession()
```

## 便于计算的函数

```
def make_kernel(a):
  """转换一个2维数组成一个卷积核"""
  a = np.asarray(a)
  a = a.reshape(list(a.shape) + [1,1])
  return tf.constant(a, dtype=1)

def simple_conv(x, k):
  """一个简单的2维卷积操作"""
  x = tf.expand_dims(tf.expand_dims(x, 0), -1)
  y = tf.nn.depthwise_conv2d(x, k, [1, 1, 1, 1], padding='SAME')
  return y[0, :, :, 0]

def laplace(x):
  """计算一个数组的拉普拉斯变换"""
  laplace_k = make_kernel([[0.5, 1.0, 0.5],
                           [1.0, -6., 1.0],
                           [0.5, 1.0, 0.5]])
  return simple_conv(x, laplace_k)
```

其中，[tf.nn.depthwise_conv2d](https://www.tensorflow.org/api_docs/python/tf/nn/depthwise_conv2d)的
图示为![](https://qiita-image-store.s3.amazonaws.com/0/25288/6f0d9da1-8f53-cf43-883a-fe55464376b8.png)

## 定义概率密度函数

假设池塘的尺寸为N*N，N=500。
```
# 池塘的初始状态，初始化为0
u_init = np.zeros([N, N], dtype=np.float32)
ut_init = np.zeros([N, N], dtype=np.float32)
# 用几个随机点表示雨点，落在池塘里
for n in range(40):
  a,b = np.random.randint(0, N, 2)
  u_init[a,b] = np.random.uniform()
# 显示效果
DisplayArray(u_init, rng=[-0.1, 0.1])
```
接下来，定义微分方程。
```
# 参数
# eps -- time resolution
# damping -- wave damping
eps = tf.placeholder(tf.float32, shape=())
damping = tf.placeholder(tf.float32, shape=())
# 创建表示仿真状态的变量
U  = tf.Variable(u_init)
Ut = tf.Variable(ut_init)
# 离散化的概率密度函数的更新规则
U_ = U + eps * Ut
Ut_ = Ut + eps * (laplace(U) - damping * Ut)
# 更新状态的操作
step = tf.group(
  U.assign(U_),
  Ut.assign(Ut_))
```

## 运行仿真
```
# 初始化所有变量
tf.global_variables_initializer().run()
# 概率密度函数，运行1000次，每一次都输出一个池塘的表面状态结果
for i in range(1000):
  # Step simulation
  step.run({eps: 0.03, damping: 0.04})
  DisplayArray(U.eval(), rng=[-0.1, 0.1])
```
最终结果![](https://www.tensorflow.org/images/pde_output_2.jpg)

参考文献：
[1]
[tf中几种卷积的介绍](http://qiita.com/YusukeSuzuki@github/items/0764d15b9d0b97ec1e16)



(https://jbt.github.io/markdown-editor/#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)
