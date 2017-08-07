# 图片识别

图片识别用于识别图片属于哪一类物体,本质是图片的分类问题(机器学习主要用于两方面,回归和分类),此处以MNIST为例以数字识别介绍图片识别.

## 1. 环境
python 2.7
tensorflow 1.2.1

## 2. 程序说明
程序使用 tensorflow branch r1.2 中的https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/examples/learn/mnist.py

$ ./minst.py
本程序采用两种方式进行训练,一种为LinearClassifier,另一种为cnn,其中cnn有进行两层卷积,下面进行详细说明.

### 2.1 LinearClassifier
```python
feature_columns = learn.infer_real_valued_columns_from_input(
      mnist.train.images)
  classifier = learn.LinearClassifier(
      feature_columns=feature_columns, n_classes=10)
  classifier.fit(mnist.train.images,
                 mnist.train.labels.astype(np.int32),
                 batch_size=100,
                 steps=1000)
  score = metrics.accuracy_score(mnist.test.labels,
                                 list(classifier.predict(mnist.test.images)))
  print('Accuracy: {0:f}'.format(score))
```
LinearClassifier较易理解,不进一步说明

### 2.2 cnn

cnn 部分主要包括卷积层,池化层和全连接层,详细介绍如下:

#### 2.2.1 池化层
```python
def max_pool_2x2(tensor_in):
  return tf.nn.max_pool(
      tensor_in, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
```

池化使用过滤器为大小为2*2, 使用max_pooling,长和宽的步长均为2,padding方式为'SAME',所以池化后图片大小变为原来的一半.

#### 2.2.2 卷积层
```python
def conv_model(feature, target, mode):
  """2-layer convolution model."""
  # Convert the target to a one-hot tensor of shape (batch_size, 10) and
  # with a on-value of 1 for each one-hot vector of length 10.
  target = tf.one_hot(tf.cast(target, tf.int32), 10, 1, 0)

  # Reshape feature to 4d tensor with 2nd and 3rd dimensions being
  # image width and height final dimension being the number of color channels.
  feature = tf.reshape(feature, [-1, 28, 28, 1])

  # First conv layer will compute 32 features for each 5x5 patch
  with tf.variable_scope('conv_layer1'):
    h_conv1 = layers.convolution2d(
        feature, 32, kernel_size=[5, 5], activation_fn=tf.nn.relu)
    h_pool1 = max_pool_2x2(h_conv1)

  # Second conv layer will compute 64 features for each 5x5 patch.
  with tf.variable_scope('conv_layer2'):
    h_conv2 = layers.convolution2d(
        h_pool1, 64, kernel_size=[5, 5], activation_fn=tf.nn.relu)
    h_pool2 = max_pool_2x2(h_conv2)
    # reshape tensor into a batch of vectors
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

  # Densely connected layer with 1024 neurons.
  h_fc1 = layers.dropout(
      layers.fully_connected(
          h_pool2_flat, 1024, activation_fn=tf.nn.relu),
      keep_prob=0.5,
      is_training=mode == tf.contrib.learn.ModeKeys.TRAIN)

  # Compute logits (1 per class) and compute loss.
  logits = layers.fully_connected(h_fc1, 10, activation_fn=None)
  loss = tf.losses.softmax_cross_entropy(target, logits)

  # Create a tensor for training op.
  train_op = layers.optimize_loss(
      loss,
      tf.contrib.framework.get_global_step(),
      optimizer='SGD',
      learning_rate=0.001)

  return tf.argmax(logits, 1), loss, train_op

```
该卷积层使用两层卷积,卷基层使用的过滤器大小为5*5,padding方式为'SAME',激活函数使用relu,损失函数为SGD;
* 第一层卷积输入为28*28,卷积后的大小为28*28,设置输出的深度为32;然后经过池化层处理后深度为32,大小为14*14;
* 第二层卷积输入为14*14,卷积后的大小为14*14,深度为64,经过池化处理后深度仍为64,大小为7*7
* 在两层卷积和池化后加一个全连接层,全连接层使用1024核.
* 在全连接后使用softmax进行分类

#### 2.2.3 cnn调用
```python
### Convolutional network
  classifier = learn.Estimator(model_fn=conv_model)
  classifier.fit(mnist.train.images,
                 mnist.train.labels,
                 batch_size=100,
                 steps=20000)
  score = metrics.accuracy_score(mnist.test.labels,
                                 list(classifier.predict(mnist.test.images)))
  print('Accuracy: {0:f}'.format(score))
```

### 2.3 运行结果
LinearClassifier方法在测试集上正确率为0.921600,cnn在测试集上正确率为0.967500


__<font color=red size=72>以上为个人理解,如有错误请指正,感谢!!!</font>__
