# keras-quick-startup
IMDB Sentiment Analysis Using Keras. Just for experience.

# 背景介绍
文本分类是机器学习中一个非常常见而且重要的问题，比如新闻出版按照栏目分类（体育，旅游，军事，科技等），还有常见的网页分类，个性化新闻智能推荐，垃圾邮件过滤，情感分析等，都是文本分类的应用场景。分类有二分类（binary）和多分类（multiple-classes）。

传统的机器学习和深度学习都可以完成文本分类的任务。如果用传统的分类算法，比如朴素贝叶斯或者SVM等算法，文本分类其实就是对文本进行特征提取，确定一个评价函数，训练分类模型，然后应用于预测的过程。

在深度学习领域，有一种循环神经网络LSTM，在实践效果上可以轻松取得比传统学习算法更好的分类结果，本文就是使用LSTM神经网络对IMDB的影评进行分类，以展示深度学习在Sequence classification分类任务上的性能。

文本可以看做是一连串文字的序列，Sequence classification就是在你有一些输入序列的情况下，预测这个序列所属的分类。细想起来，这其实是个很难的问题。原因如下：序列在长度上可能差异很大。电影评论有的很长，有的只有寥寥数语。组成这个序列的输入文字又是来自一个非常大的词表。据统计常用的英文单词有3000个，常用的中文词有5000个。这需要模型去学习比较长的上下文。

在深度学习领域有一个比较适合于解决这个场景的循环神经网络，那就是长短时记忆网络(Long Short Term Memory Network, LSTM)，它较好地解决了原始循环神经网络的缺陷，成为当前最流行的RNN，在语音识别、图片描述、自然语言处理等许多领域中成功应用。关于LSTM网络，推荐看中开社人工智能大牛hanbingtao的这篇博客：[零基础入门深度学习(6) - 长短时记忆网络(LSTM)](https://www.zybuluo.com/hanbingtao/note/581764)，这是一系列讲解深度学习的博客，清晰而且系统。

# 问题定义
使用广为熟知的IMDB电影评论来做情感分类。这个庞大的影评数据集包含了25000 个影评 (good or bad) 用于训练（Train），同样还有25000个评论用于测试（Test）。我们要做的事情就是给定一条电影评论，判断它的情感是正面的（good）还是负面的（bad）。这些数据是由Stanford学者收集的，2011年的论文中，将数据集对半分成训练集和测试集，当时取得了88.89%的准确率。

# 准备工具
深度学习领域有很多优秀的平台，比如TensorFlow、MXNet、Torch、Caffe等。[TensorFlow](https://github.com/tensorflow/tensorflow)是Google推出的可扩展的深度学平台，可以完成基于data flow graphs的计算任务。使用过TensorFlow API进行编程的同学可能感觉到TensorFlow提供的API虽然功能非常强大，但是抽象的程度比较低，比较冗长和繁琐，使用起来不是很自然。Keras是一个高层神经网络API，由纯Python编写而成并基Tensorflow或Theano。Keras 能够把你的idea迅速转换为结果，非常适合于简易和快速的原型设计，支持CNN和RNN，或二者的结合；无缝CPU和GPU切换。Keras适用的Python版本是：Python 2.7-3.5。更详细的信息请参考[Keras中文文档](http://keras-cn.readthedocs.io/en/latest/)。本文使用Keras来完成这个分析任务，代码更简洁易读。

# 构建环境
推荐使用docker镜像的方式搭建深度学习平台环境。不过说实话，深度学习没有GPU或者强悍的服务器，用起来还真是不容易，作为一个简单的教程，我们还是使用CPU模式方便大家都能用起来。最基础的软件栈，需要一套python开发环境，Keras和Tensorflow的最新版本。现在构建环境已经是非常便捷了，不用从头开始构建镜像。store.docker.com 上面已有很多现成的镜像可以使用，选择一个合适的基础镜像，根据自己的需求进行修改，构建适应自己环境的镜像文件。

笔者选择一个只有命令行模式的keras镜像，通过添加jupyter-notebook 来创建一个更适合编码的镜像环境。

> 如果你还没有接触过Jupyter Notebook，这里简单说明一下。Jupyter notebook 此前被称为IPython notebook，是一个交互式笔记本，支持运行 40 多种编程语言。对于希望编写漂亮的交互式文档的人来说是一个强大工具。同时支持代码和Markdown格式的文档，方便分享代码。很多主流的代码仓库或者会议，都会以Jupyter Notebook的方式进行对外交流。

构建docker镜像的Dockerfile文件如下：

```
FROM gw000/keras:2.0.4-py2-tf-cpu

RUN pip install \
    jupyter \
    matplotlib \
    seaborn

VOLUME /notebook
WORKDIR /notebook
EXPOSE 8888
CMD ["jupyter", "notebook", "--allow-root", "--no-browser", "--ip=0.0.0.0", "--NotebookApp.token="]

```
Docker镜像构建指令：`docker build -t keras:2.0.4-py2-tf1.1.0-cpu .`

我们采用TensorFlow作为Keras的backend，本镜像包含的主要组件版本号如下：

```
Keras (2.0.4)
numpy (1.12.1)
pandas (0.20.1)
tensorflow (1.1.0)
```

使用`docker run`命令启动一个容器：`docker run -d --name keras -v /sharedfolder:/notebook/sharedfolder -p 8888:8888 keras:2.0.4-py2-tf1.1.0-cpu`

容器启动完成之后，就可以通过浏览器访问到jupyter notebook的编程页面了。而且在容器和主机之间做了文件映射，方便共享。建议使用chrome或者firefox浏览器来访问，具体访问地址就是`http://${your-container-host-ip}:8888`

# 数据理解
Keras提供了直接访问IMDB dataset的内置函数 `imdb.load_data()` 。调用该函数能够直接生成深度学习模型的数据格式。评论中的单词已经被一个整形值取代，这个整形值代表了这个单词在数据集中的索引。因此影评序列就是由一连串的整形值组成。

# 词嵌入 Word Embedding
我们将把每一条影评都映射到一个实值向量空间。这个技术在文本处理过程中被称作词嵌入。经过词嵌入之后，单词被编码成高维空间的一个实值向量，在含义上比较接近的单词在向量空间中也是比较接近的（cosine距离）。

Keras提供了一个Embedding layer非常方便实现这个过程。我们将每一个单词都映射为一个32维的实值向量。同时限制我们建模中用到的总的单词数取前5000个最常出现的单词。其他的都以0代替。
由于评论中序列的长度长短不一，我们限制每个评论最多有500个单词，超过500的截断，小于500的用0填充。
这样我们数据集的输入数据的表示方式都定义好了。下面就着手定义LSTM网络的结构。

# 一个简单的LSTM 序列分类网络
快速设计一个LSTM神经网络模型，看看分类的性能如何。上文keras:2.0.4-py2-tf1.1.0-cpu这个镜像中已经包含了Keras和TensorFlow以及必要的依赖，直接启动容器，通过Jupyter Notebook的web界面编程即可。

> 下文有完整的实现代码，文中会对不同的模型结构递进式分析调优。由于代码的主干是一样的，所以下文第一个样例代码，分段表述，对代码的功能进行详细分析。后面的代码只是有少许不同，就不分段详细解释了。只是把关键的建模代码添加注释。

```python
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# 为了能够重现我们的结果，固定一个随机数种子。
numpy.random.seed(7)
```

获取数据集，数据集中50%用于训练，50%用于测试。使用top5000单词。
> 备注：为什么选择top5000？莫斯科国立语言研究所通过对英国、美国、法国、西班牙的34部文学作品的分析和研究，前5000个单词占到了93.5%。顺便也说一下在一级和二级国标中，常用汉子为7000个，因此如果用于中文的文本序列分析，可以取top7000。


```python
# 加载imdb数据集，使用top5000单词，其他reset为0
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

```

接下来对输入序列进行截断和填补，以保持相同的长度。模型会学习到填补的0值不包含任何信息。因此，虽然输入序列的长度可能不一样，但是在计算的时候需要保持向量的长度是一致的。

```python
max_review_length = 500
X_train=sequence.pad_sequences(X_train, maxlen = max_review_length)
X_test=sequence.pad_sequences(X_test, maxlen = max_review_length)
```

现在我们可以来定义和训练LSTM模型了。

+ 第一层是词嵌入Embedded layer，使用32维向量来表达每一个单词。
+ 接下来是LSTM layer，包含100个记忆单元。 
+ 最后，因为是一个二分类问题，我们使用一个Dense output layer，使用sigmoid激活函数输出0或者1的二分类结果。
+ 优化器optimizer选用adam算法，当然也可以选择其他算法，如sgd。
+ 二分类问题，我们使用binary_crossentropy作为损失函数。
+ 选择一个较大的batch_size=64

```python
embedding_vector_length=32
model=Sequential()
model.add(Embedding(top_words,embedding_vetcor_length,input_length=max_review_length))
model.add(LSTM(100)) # LSTM的具体用法参考 https://keras.io/layers/recurrent/#lstm
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=5,batch_size=64)
```


    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 500, 32)           160000    
    _________________________________________________________________
    lstm_1 (LSTM)                (None, 100)               53200     
    _________________________________________________________________
    dense_1 (Dense)              (None, 1)                 101       
    =================================================================
    Total params: 213,301
    Trainable params: 213,301
    Non-trainable params: 0
    _________________________________________________________________
    None
    Train on 25000 samples, validate on 25000 samples
	Epoch 1/5
	25000/25000 [==============================] - 229s - loss: 0.4756 - acc: 0.7594 - val_loss: 0.3335 - val_acc: 0.8612
	Epoch 2/5
	25000/25000 [==============================] - 227s - loss: 0.2982 - acc: 0.8809 - val_loss: 0.3328 - val_acc: 0.8651
	Epoch 3/5
	25000/25000 [==============================] - 228s - loss: 0.2415 - acc: 0.9058 - val_loss: 0.3282 - val_acc: 0.8742
	Epoch 4/5
	25000/25000 [==============================] - 228s - loss: 0.2186 - acc: 0.9160 - val_loss: 0.3354 - val_acc: 0.8631
	Epoch 5/5
	25000/25000 [==============================] - 227s - loss: 0.1865 - acc: 0.9297 - val_loss: 0.3733 - val_acc: 0.8614

    <keras.callbacks.History at 0x7f0f3a56ff90>


模型训练完成之后，我们测试一下模型的性能。测试集上的模型性能质保，其实就是最后一个Epoch的 **val_acc** 指标。

```python
# evaluation
scores=model.evaluate(X_test,y_test,verbose=0)
print("Accuracy: %.2f%%"%(scores[1]*100))
```
    Accuracy: 86.14%

# 回顾这个LSTM模型
这是一个使用LSTM做文本分类的常规流程，可以作为解决其他序列分类问题的模板。
这个网络只是用了基本的输入数据预处理，没有多少的调优，在第二个Epoch已经取得了86.51%的准确率，分类效果已经比较好了，第三个Epoch在训练集和测试集上继续性能继续提升。

我们还要意识到，神经网络模型容易出现过拟合**overfitting**问题，出现过拟合的时候，虽然loss训练误差在每个Epoch稳步下降，但是在测试集上**val_loss没有下降，反而有上升的趋势**，如果val_loss比loss高出很多，这说明模型已经严重过拟合了。下面我们就要考虑解决过拟合这个问题。

过拟合的问题是搞机器学习的人绕不开的话题，我们无法在取得较高精度的情况下，又能避免过拟合问题，所以需要在模型拟合测试集（及我们案例中的尽量减少loss，获得较高的acc）与模型的泛化能力（val_loss也比较小，与loss相差不大，说明泛化能力较好）之间做一个tradeoff。在深度学习领域，Dropout Layer可以作为减少过拟合风险的一种技术。我们看看在添加Dropout Layer之后的模型性能。

## 使用Dropout
添加Dropout Layers的代码样例:

```python
# LSTM with Dropout
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

numpy.random.seed(7)
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

# truncate and pad
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

# create model
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(top_words,embedding_vector_length,input_length=max_review_length))
model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.1))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=5, batch_size=64)
# evaluation
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
```

    Using TensorFlow backend.


    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 500, 32)           160000    
    _________________________________________________________________
    lstm_1 (LSTM)                (None, 100)               53200     
    _________________________________________________________________
    dense_1 (Dense)              (None, 1)                 101       
    =================================================================
    Total params: 213,301
    Trainable params: 213,301
    Non-trainable params: 0
    _________________________________________________________________
    None
    Train on 25000 samples, validate on 25000 samples
	Epoch 1/5
	25000/25000 [==============================] - 241s - loss: 0.5111 - acc: 0.7414 - val_loss: 0.4154 - val_acc: 0.8124
	Epoch 2/5
	25000/25000 [==============================] - 241s - loss: 0.3759 - acc: 0.8379 - val_loss: 0.3752 - val_acc: 0.8360
	Epoch 3/5
	25000/25000 [==============================] - 240s - loss: 0.4841 - acc: 0.7906 - val_loss: 0.3842 - val_acc: 0.8358
	Epoch 4/5
	25000/25000 [==============================] - 241s - loss: 0.3394 - acc: 0.8605 - val_loss: 0.3427 - val_acc: 0.8573
	Epoch 5/5
	25000/25000 [==============================] - 241s - loss: 0.2961 - acc: 0.8798 - val_loss: 0.3361 - val_acc: 0.8647  
    Accuracy: 86.47%

我们从执行结果看，Accuracy与没有Dropout的时候第二个Epoch还要降低一些，但是由于减少了过拟合的风险，模型的结构风险会降低，迁移能力理论上会好一些。

加入Dropout层之后，调整Dropout参数，可以减少过拟合的风险，不过这个超参数的设置需要经验，或者说要多尝试几次。但是仍然无法避免过拟合现象。Keras提供了一个回调函数EarlyStopping()，可以针对Epoch出现val_acc降低的时候，提前停止训练，可以参考keras的官方文档：[keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')](https://keras.io/callbacks/#earlystopping) 试试看。通常在工程实践中，我们可以认为如果模型在测试集上连续的5个epoch中的性能表现都没有提升，则认为可以提前停止了。


解决过拟合问题，目前有很多手段，比如调小学习速率，调小反向传播的训练样本数（batch_size）都可能减少过拟合的风险，但是这里面的小tricks有点说不清道不明，还可以试一试别的optimizer，哪个好选哪个，是不是感觉像碰运气。机器学习里面有很多经验性的东西是要多试试才能感受到。解决过拟合问题还有一个重要的手段是使用正则化技术。

```python
from keras import regularizers
model.add(Dense(64, input_dim=64,
                kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))
```                
在keras中，可用的正则化方法有如下三个：

```python
keras.regularizers.l1(0.)
keras.regularizers.l2(0.)
keras.regularizers.l1_l2(0.)
```

使用正则化控制过拟合的实验代码如下：
```python
model = Sequential()
model.add(Embedding(top_words,embedding_vector_length,input_length=max_review_length))
model.add(LSTM(100)) 
model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.001)))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=10, batch_size=64)
```

	Layer (type)                 Output Shape              Param #   
	=================================================================
	embedding_28 (Embedding)     (None, 500, 32)           160000    
	_________________________________________________________________
	lstm_28 (LSTM)               (None, 100)               53200     
	_________________________________________________________________
	dense_27 (Dense)             (None, 1)                 101       
	=================================================================
	Total params: 213,301
	Trainable params: 213,301
	Non-trainable params: 0
	_________________________________________________________________
	None
	Train on 25000 samples, validate on 25000 samples
	Epoch 1/10
	25000/25000 [==============================] - 235s - loss: 0.6545 - acc: 0.6708 - val_loss: 0.6525 - val_acc: 0.7307
	Epoch 2/10
	25000/25000 [==============================] - 234s - loss: 0.5124 - acc: 0.7773 - val_loss: 0.5140 - val_acc: 0.7730
	Epoch 3/10
	25000/25000 [==============================] - 233s - loss: 0.4345 - acc: 0.8295 - val_loss: 0.4061 - val_acc: 0.8470
	Epoch 4/10
	25000/25000 [==============================] - 235s - loss: 0.3456 - acc: 0.8804 - val_loss: 0.4144 - val_acc: 0.8472
	Epoch 5/10
	25000/25000 [==============================] - 234s - loss: 0.3122 - acc: 0.8976 - val_loss: 0.3772 - val_acc: 0.8606
	Epoch 6/10
	25000/25000 [==============================] - 235s - loss: 0.2816 - acc: 0.9112 - val_loss: 0.3769 - val_acc: 0.8680
	Epoch 7/10
	25000/25000 [==============================] - 233s - loss: 0.2782 - acc: 0.9115 - val_loss: 0.3719 - val_acc: 0.8664
	Epoch 8/10
	25000/25000 [==============================] - 234s - loss: 0.2589 - acc: 0.9192 - val_loss: 0.3831 - val_acc: 0.8710
	Epoch 9/10
	25000/25000 [==============================] - 233s - loss: 0.2416 - acc: 0.9275 - val_loss: 0.3677 - val_acc: 0.8724
	Epoch 10/10
	25000/25000 [==============================] - 233s - loss: 0.2440 - acc: 0.9261 - val_loss: 0.3717 - val_acc: 0.8630

# 结合卷积神经网络优化序列分类的整体性能

卷积神经网络在图像识别和语音识别领域取得了非凡的成就，特别擅长于从输入数据中学习spatial structure，因此对于处理NLP问题很有帮助。利用CNN学习到的特征可以用于LSTM层的训练，对模型的性能理论上会有提升。
我们使用keras可以很方便的在Embedding Layer添加一个 one-dimensional CNN and max pooling layers。以此作为LSTM层的特征输入。

代码示例如下:

```python
# LSTM with Dropout and CNN classification
embedding_vector_length=32
model=Sequential()
model.add(Embedding(top_words,embedding_vector_length,input_length=max_review_length))
model.add(SpatialDropout1D(0.3))
model.add(Conv1D(activation="relu", padding="same", filters=64, kernel_size=5))
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(100,dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())
model.fit(X_train,y_train,validation_data=(X_test,y_test), epochs=10,batch_size=64)
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_3 (Embedding)      (None, 500, 32)           160000    
    _________________________________________________________________
    spatial_dropout1d_1 (Spatial (None, 500, 32)           0         
    _________________________________________________________________
    conv1d_3 (Conv1D)            (None, 500, 32)           3104      
    _________________________________________________________________
    max_pooling1d_3 (MaxPooling1 (None, 250, 32)           0         
    _________________________________________________________________
    lstm_3 (LSTM)                (None, 100)               53200     
    _________________________________________________________________
    dense_3 (Dense)              (None, 1)                 101       
    =================================================================
    Total params: 216,405
    Trainable params: 216,405
    Non-trainable params: 0
    _________________________________________________________________
    
	Train on 25000 samples, validate on 25000 samples
	Epoch 1/10
	25000/25000 [==============================] - 84s - loss: 0.4190 - acc: 0.7912 - val_loss: 0.2779 - val_acc: 0.8865
	Epoch 2/10
	25000/25000 [==============================] - 81s - loss: 0.2685 - acc: 0.8924 - val_loss: 0.2685 - val_acc: 0.8908
	Epoch 3/10
	25000/25000 [==============================] - 81s - loss: 0.2209 - acc: 0.9156 - val_loss: 0.2739 - val_acc: 0.8895
	Epoch 4/10
	25000/25000 [==============================] - 81s - loss: 0.1917 - acc: 0.9264 - val_loss: 0.2909 - val_acc: 0.8817
	Epoch 5/10
	25000/25000 [==============================] - 81s - loss: 0.1728 - acc: 0.9349 - val_loss: 0.3080 - val_acc: 0.8825
	Epoch 6/10
	25000/25000 [==============================] - 81s - loss: 0.1553 - acc: 0.9428 - val_loss: 0.3064 - val_acc: 0.8829
	Epoch 7/10
	25000/25000 [==============================] - 81s - loss: 0.1365 - acc: 0.9508 - val_loss: 0.3297 - val_acc: 0.8790
	Epoch 8/10
	25000/25000 [==============================] - 81s - loss: 0.1283 - acc: 0.9537 - val_loss: 0.3192 - val_acc: 0.8844
	Epoch 9/10
	25000/25000 [==============================] - 81s - loss: 0.1101 - acc: 0.9613 - val_loss: 0.3639 - val_acc: 0.8813
	Epoch 10/10
	25000/25000 [==============================] - 81s - loss: 0.1026 - acc: 0.9644 - val_loss: 0.3803 - val_acc: 0.8812

添加CNN layer之后，每一轮的训练时间大大减少了，大约降到了原来的1/3时间，精度也有所提升，整体上取得了更好的performance。

# 总结回顾
这篇文章我们介绍了如何用LSTM网络来解决文本分类问题。如何减少模型过拟合的风险，以及怎样结合CNN网络中学习到的spatial structure来优化NLP问题的特征，从而提升整个网络的性能。
对于一个文本分类问题，我们可以沿着这个思路设计我们的网络结构，基本上能应该能够解决常见的文本序列分类问题了。当然如果要在整个基础上继续小步提升，还需要对数据进行较多的预处理，对网络的参数进行经验性改进。
