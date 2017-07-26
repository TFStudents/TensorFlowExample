# Mandelbrot Set

在此部分，介绍一个使用tensorflow可视化Mandelbrot set的例子。以这个例子表明，tensorflow可用于基础数学。以下内容来源于tensorflow教程部分[Mandelbrot Set](https://www.tensorflow.org/tutorials/mandelbrot)，对其进行了翻译和整理。

目录:

* 什么是Mandelbrot set
* 可视化Mandelbrot set

## 什么是Mandelbrot set（曼德博集合）？

Mandelbrot set是在[复平面](https://zh.wikipedia.org/wiki/%E5%A4%8D%E5%B9%B3%E9%9D%A2)组成[分形](https://zh.wikipedia.org/wiki/%E5%88%86%E5%BD%A2)的点的集合，以美国数学家Mandelbrot命名。

Mandelbrot set是一个复数点c的集合，任取复平面上的一个点c，若满足：Z(n+1) = Z(n)^2 + c，Z(0)=0，当n不断增大时，|Z(n)|有界，则c属于Mandelbrot set。

在实际中，一般取一个较大的n值，迭代计算Z(n)是否有界（小于某个值），若有界则c属于Mandelbrot set。在可视化的时候，可选择在平面上c点的坐标以某种 颜色显示。

## 可视化Mandelbrot set

### import相关的库

导入仿真计算的库

```
import tensorflow as tf
import numpy as np
```

导入可视化的库
```
import PIL.Image
from io import BytesIO
from IPython.display import Image, display
```

### 定义一个可视化函数

```
def DisplayFractal(a, fmt='jpeg'):
  a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])
  img = np.concatenate([10+20*np.cos(a_cyclic),
                        30+50*np.sin(a_cyclic),
                        155-80*np.cos(a_cyclic)], 2)
  img[a==a.max()] = 0
  a = img
  a = np.uint8(np.clip(a, 0, 255))
  f = BytesIO()
  PIL.Image.fromarray(a).save(f, fmt)
  display(Image(data=f.getvalue()))
 
```

### 使用tensorflow进行一些数学计算

```
sess = tf.InteractiveSession() # 打开一个交互式的session，在这个session下进行计算

Y, X = np.mgrid[-1.3:1.3:0.005, -2:1:0.005] # 创建复平面上点的横纵坐标
Z = X+1j*Y # 复数Z

xs = tf.constant(Z.astype(np.complex64))  # xs为tf的常量，其值为Z转成np.complex64类型后的值
zs = tf.Variable(xs) # zs为tf中的变量，其初始值为xs
ns = tf.Variable(tf.zeros_like(xs, tf.float32)) # ns为tf中的变量，与xs的shape相同，但值都为0，tf.float32类型。

tf.global_variables_initializer().run() # 计算前先初始化所有变量

zs_ = zs*zs + xs # 对应Mandelbrot set的迭代计算公式

not_diverged = tf.abs(zs_) < 4 # 小于4时即收敛，与zs_的shape相同，但所有值为bool型

step = tf.group(
  zs.assign(zs_), # 将zs_赋值给zs
  ns.assign_add(tf.cast(not_diverged, tf.float32)) # ns = ns+not_diverged
  )
  
 for i in range(200): 
     step.run() # 迭代200次
  
```

### 可视化结果

```
DisplayFractal(ns.eval()) # 以上代码在ipython中运行时，并未出现可视化的图片。在jupyter notebook中，出现可视化的结果。
```
![](https://www.tensorflow.org/images/mandelbrot_output.jpg)


（https://jbt.github.io/markdown-editor/#fVNdbxtFFH2fX3GrVHLcJl54DU8oFTRSIiFSniKkjHdnvWPv7l3NXMdNEVLERylqKEilUJqKKhIfEVJSJF7SuCE/hl3bPPUvcGfWOClqkbJaZ+bcc+459+4crMk8UmnbIMG6IiGqxwejw5/+/vSgunP7xfPdcnh3PPyqPN4pj38rT8/G3x6Qyi2aOMVB9fXTya+3q93vLnBYReNHn5V/3q0Ov/lr55Ny+PPk7AdX608m+wejh/eY9iUSJi1P7lVPTsb7O6MHv1eHv3ClEOO9o+r0wZIQV6Ac7pTP7owePn1ZiW9e04MQc3Ovq3rx/IkQHvA/xXOgswINjfeOq8//YEvVyf36vDp6VD77sk5kRlB9ccqd14D/xjQ525vs77qCk73a3uRof3z0/ZTuX4rx8P7ox8dCwBVYJpNeXYcAlrOI34Rg5ZYCShTEOlXnkETHdA6c/sfwMEG0alYoLY/Z9CIc5IAGrt9YW3Uc14zsADuHyGAB0nODzrkqUcZXpygj0OTA77jLEHP2RRYk3ztqvs19Xx+8vwoWYRv7EMocbOIQjtBy1CuNDHKEDmIEkmBgNOm8wwRZwaSBw6XKAKmbtOBYOugxLIGZosRhmdhYlcYtIVYRewvcbqotXfJ5xYju1Zamft0S4m225Vw0rCfhziN1CZauvsn7tLm52ZVb0oZGFyQuz8f9PCSN+XzzIwFweb4R6a1Gs5VQls43VkBmLMZHrUbzLfExP1wvxI1EW+C/DczhXU3X++0P5xOiwi4FQUdT0m+3QsyCbpuCbJr9ooo0oWk6i6ki4LZ6OQ5Ax7DS4DG13zA9lyj5lgeufbYr3uPpWDeNNdOCa4htPzMnv+FcQc1aq7N4ZiLGePHCYFeFZAOHm6oHzQWIDWYwSHSYCDc7ndtCG+kicDIcuF3wGj65hH9tu68hVRnPvobxTIhBocxci26P16kfx9C3bIA5MtlTnmjJD2hjFoGmV8Z04f7i7yaP1pzvbiGN5WXwjMtsaU0bc8G5c5n5o1auKKiLnUE5UN6K3eb+by4mupOk/BD3WofiGWfHra6dcVqMacC7nMlcy9C20HT8WTBDByqfStX0MLtxe8ufB/ap6JPfQGinGPasl+vaxUjFqST1ykQimfewI3VwjqtVOrd0UThqjCGSJGdx897E/LAif4xW/AM=）



 参考文献：
 
 [1] [曼德布洛特集合简要概述](https://msdn.microsoft.com/zh-cn/library/jj635753(v=vs.85).aspx)
 
 [2] [tensorflow中的Mandelbrot set教程](https://www.tensorflow.org/tutorials/mandelbrot)
 
 
 

