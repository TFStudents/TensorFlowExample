# Tensorflow Wide & Deep Learning Tutorial

此部分，简单介绍了一个使用tensorflow在一份数据集上训练wide & deep模型，并使用训练好的模型进行预测的例子。全部内容基于[tf官网例子](https://www.tensorflow.org/tutorials/wide_and_deep)进行了整理。tf官网上该例子的[源代码](https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/examples/learn/wide_n_deep_tutorial.py)，请结合源代码看此部分教程。

## 涉及到：
* 数据集：[Census Income Data](https://archive.ics.uci.edu/ml/datasets/Census+Income)
* 模型：Wide & Deep Model [论文](https://arxiv.org/abs/1606.07792)，[理论介绍](http://www.shuang0420.com/2017/03/13/%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0%20-%20Wide%20and%20Deep%20Learning%20for%20Recommender%20Systems/)
* pandas（python的一个数据处理库，在这里用于读取数据集）
* tempfile（python的一个临时文件相关的库，在这里用于创建临时文件，关闭后会自动删除）
* [tf.contrib.learn.DNNLinearCombinedClassifier](https://www.tensorflow.org/api_docs/python/tf/contrib/learn/DNNLinearCombinedClassifier)（Wide & Deep Model在tensorflow中的实现）
* [tf.contrib.layers.sparse_column_with_keys](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/sparse_column_with_keys)（tf中处理分类型特征的一种方式，当知道该特征中不同的特征值时使用）
* [tf.contrib.layers.sparse_column_with_hash_bucket](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/sparse_column_with_hash_bucket)（tf中处理分类型特征的另一种方式，如果不知道该特征中不同的特征值时使用，但特征值必须时字符串或整型）
* [tf.contrib.layers.real_valued_column](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/real_valued_column)（tf中处理连续型特征的一种方式）
* [tf.contrib.layers.bucketized_column](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/bucketized_column)（tf中离散化连续型特征的一种方式）
* [tf.contrib.layers.crossed_column](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/crossed_column)（tf中处理交叉特征的一种方式）
* [tf.contrib.layers.embedding_column](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/embedding_column)（tf中embedding稀疏特征，并feed给DNN）

## 目录：
* Census Income Data介绍和下载
* Wide & Deep Model简介
* 数据预处理
* 模型的训练和评估

！！！请注意！！！为避免对有些专有名词翻译的不准确，在这里将不翻译。

## 1、 Census Income Data介绍和下载

### 1.1、 Census Income Data介绍

数据来源于美国1994年人口调查数据库。[数据集介绍](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names)。

样本个数：train=32561, test=16281。
类别：>50K, <=50K。
特征个数：14个，其中：
分类型特征：CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",
                       "relationship", "race", "gender", "native_country"]
 连续型特征：CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss",
                      "hours_per_week"]
 
 ### 1.2、Census Income Data下载
 
 1） 手动下载，[数据集地址](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/)
 
 2） 使用[源代码](https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/examples/learn/wide_n_deep_tutorial.py)中的maybe_download函数下载
 
 ### 2、Wide & Deep Model简介
 
 ![Wide & Deep Model](https://www.tensorflow.org/images/wide_n_deep.svg)
 
上图左为Wide Model；上图右为Deep Model；上图中为Wide & Deep Model，该模型由两个模型组成：Wide Model（线性模型），Deep Model（DNN模型）。在tensorflow中有该模型的实现：[tf.contrib.learn.DNNLinearCombinedClassifier](https://www.tensorflow.org/api_docs/python/tf/contrib/learn/DNNLinearCombinedClassifier)（模型的理论介绍，可从本篇开头介绍的链接找到）。DNNLinearCombinedClassifier函数的参数有linear_feature_columns，dnn_feature_columns，dnn_hidden_units等。其中linear_feature_columns为模型的wide part的输入，包括sparse features和sparse crossed features（交叉特征）；dnn_feature_columns为模型的deep part的输入，包括连续型特征，每一个分类型特征的embedding。



### 3、数据预处理

1） 原数据集中含有缺失值，而模型的输入不能含有缺失值，需去除含有缺失值的样本。在[源代码](https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/examples/learn/wide_n_deep_tutorial.py)的train_and_eval函数中有相应处理，利用了pandas的dropna函数。

我手动下载了数据集，发现其中的缺失值用'?'代替了。我的处理方案时，使用pd.read_csv读取数据集，返回dataframe对象，设为df。

l = df.apply(lambda x:'?' in x.values, axis=1) # 如果某一行含有'?'，则为True，否则为False。
df1 = df.drop(df.loc[l].index) # 利用得到的l来索引df，得到含有'?'的行，然后去除这些行，得到df1。

2）对于分类型特征，由于其值本身没有任何意义，一般使用OneHotEncoder来处理。在tensorflow中有相应的函数[tf.contrib.layers.sparse_column_with_keys](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/sparse_column_with_keys)，[tf.contrib.layers.sparse_column_with_hash_bucket](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/sparse_column_with_hash_bucket)，将分类型特征转成sparse features，这里并没有执行，只是先进行了定义，用于feed给DNN，下面类似的几个函数也是进行类似的定义。

将若干个分类型特征组成交叉特征的函数为[tf.contrib.layers.crossed_column](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/crossed_column)，特征工程的一种操作。

将两个类别转成0和1表示的方法，见源代码中的177和178行。

对于有些特征值很多的特征，虽然可将其转成Sparse features，但维度高，还需使用[tf.contrib.layers.embedding_column](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/embedding_column)对其降维转成embedding vecotr，才能作为wide part的输入。对于降维后其维度的确定，可参考 
[官网教程的The Deep Model: Neural Network with Embeddings部分](https://www.tensorflow.org/tutorials/wide_and_deep)，设置为log2(n)或者k*(n)**(1/4)，n为该特征不同的特征值的个数，k为一个小的常数，通常小于10。

 3）对于连续型特征，使用了函数[tf.contrib.layers.real_valued_column](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/real_valued_column)来处理。
 
 4）[源代码](https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/examples/learn/wide_n_deep_tutorial.py)的input_fn函数，将传入pandas读取的dataframe对象df的每一列，转成tensorflow中的tensor，并创建特征名与其tensor的Python字典。

对于连续型特征，使用tf.constant(df[特征名].values)转成一维的tensor；

对于分类型特征，使用[tf.SparseTensor](https://www.tensorflow.org/api_docs/python/tf/SparseTensor)进行转换。该函数中最重要的三个参数为indice（二维数组，表示dense feature中非0值的下标），values（一维数组，表示feature的非0值），dense_shape（一维数组，表示dense feature的shape）。

（个人观点：从源代码中tf.SparseTensor的使用来看，对于传递给该函数的每一个分类型特征，indice=[..., [i,0], ...]； value=一个dataframe列的所有值，一维数组；dense_shape=[该列的长度，1]。如果将其转成dense tensor，结果与该列的唯一区别就是维度从一维，变成了二维的[该列的长度，1]。这样的话，并没有达到sparse的特点，请高人指点。）
 
 
 ### 4、 模型的训练和评估
 
 在[源代码](https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/examples/learn/wide_n_deep_tutorial.py)中，build_estimator函数用于创建wide& deep model，返回模型的对象m。
 
 然后调用模型的[fit方法](https://www.tensorflow.org/api_docs/python/tf/contrib/learn/DNNLinearCombinedClassifier)，传入数据、迭代次数等参数进行训练。
 
 调用模型的[evaluate方法](https://www.tensorflow.org/api_docs/python/tf/contrib/learn/DNNLinearCombinedClassifier)进行预测。

（https://jbt.github.io/markdown-editor/#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）
