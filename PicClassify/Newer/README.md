# 直接运行程序
我们先不用看概念，不必知道里面的原理，直接将程序运行起来，看看效果，感受一下手写字体的识别。
1. ### 训练数据的下载   
> [地址](http://yann.lecun.com/exdb/mnist/) 下载如图绿色框内的4个数据，请确保是源文件，不要修改后缀名称  
![图片](https://github.com/TFStudents/TF/blob/master/Resource/%E8%AE%AD%E7%BB%83%E9%9B%86.png)     
在本地创建一个文件夹如MNIST_TEST（最好和代码同级目录）,将下载的4个文件放在其中。
2. ### 运行py代码  
> [mnist_softmax.py源码下载](...)  
> 将此源码下载保存到刚才新建的MNIST_TEST 同级目录下。在命令行，直接运行命令
 ```
 python mnist_softmax.py 
 ```
在ipyhon控制台会有如下类似log
```
runfile('C:/WorkspaceOther/mnist_softmax.py', wdir='C:/WorkspaceOther')
Extracting MNIST_TEST\train-images-idx3-ubyte.gz
Extracting MNIST_TEST\train-labels-idx1-ubyte.gz
Extracting MNIST_TEST\t10k-images-idx3-ubyte.gz
Extracting MNIST_TEST\t10k-labels-idx1-ubyte.gz
0.9192
```
使用ipython命令行，可能有如下错误，这是由ipython自身问题，可以忽略。
```
An exception has occurred, use %tb to see the full traceback.

SystemExit

C:\DevelopTools\Anaconda\envs\tensorflow\lib\site-packages\IPython\core\interactiveshell.py:2870: UserWarning: To exit: use 'exit', 'quit', or Ctrl-D.
  warn("To exit: use 'exit', 'quit', or Ctrl-D.", stacklevel=1)
```
# 解释代码及原理
> 直接根据就运行顺序开始走起，引包就不讲了 
- ### 训练集文件地址
```python 
FLAGS = None
DEFAULT_PATH="MNIST_TEST"
```
由于源文件和训练集是同级目录，所以只需“MNIST_TEST”，如果不在，则这里填写训练集路径就可以了

- ### 读取参数并启动main
```
if __name__ == '__main__':
  parser = argparse.ArgumentParser() #就是对命令行的参数进行解析及添加
  parser.add_argument('--data_dir', type=str, default=DEFAULT_PATH,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed) #启动main(_)函数
```
[argparse.ArgumentParser更多知识参考](https://docs.python.org/2/howto/argparse.html)     
 
 - ### 读取数据
 ```
   mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
 ```
 这才是真正的读取数据，tf已经封装了，其实FLAGS.data_dir的值为训练集的文件夹即"MNIST_TEST"，如果本地为空，系统会自动帮你下载，感兴趣可以看看源码。  
### 至于数据集本身，是由一定像素的如28x28的手写数字图片，转换而成的特殊数据类型，至于怎么制作这样的图片，制作属于自己的数据集，这是个问题，再另外的章节再一起探讨。我有这样的计划，再每学一个实例的时候，先把基本的流程及核心原理弄懂，然后对应疑惑的问题进行整理，后面再深入的研究探讨。并且在章节的末尾会留下问题，如果你对某个问题感兴趣，可以提交出来，让大家学习参考。*
 
