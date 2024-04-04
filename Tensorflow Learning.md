# Tensorflow Learning



Tensorflow Official recommendations: https://www.tensorflow.org/resources/learn-ml/basics-of-machine-learning?hl=zh-cn



## 1.TensorFlow 的机器学习基础知识

### Step 1：了解什么是机器学习

#### Books

[《使用 Python 进行深度学习》](https://www.manning.com/books/deep-learning-with-python-second-edition) 1-4章

#### Courses

[TensorFlow 简介](https://www.coursera.org/learn/introduction-tensorflow)

[TensorFlow 在深度学习中的应用简介](https://www.udacity.com/course/intro-to-tensorflow-for-deep-learning--ud187)

[神经网络工作原理](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&hl=zh-cn)



### Step 2：基础知识延伸

[《TensorFlow 开发者》专项课程](https://www.coursera.org/specializations/tensorflow-in-practice)



### Step 3：实践

[TensorFlow 核心教程](https://www.tensorflow.org/tutorials?hl=zh-cn)

Kaggle competations



### Step 4：更加深入地了解 TensorFlow

[《使用 Python 进行深度学习》](https://www.manning.com/books/deep-learning-with-python-second-edition) 5-9章

[《使用 Scikit-Learn、Keras 和 TensorFlow 进行机器学习实践》](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)



## 2.使用 TensorFlow 进行理论机器学习和高级机器学习

https://www.tensorflow.org/resources/learn-ml/theoretical-and-advanced-machine-learning?hl=zh-cn

### 第 1 步：复习数学概念

[线性代数的本质](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&hl=zh-cn)和[微积分的本质](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr&hl=zh-cn)

[线性代数](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/)或[单变量微积分](https://ocw.mit.edu/courses/mathematics/18-01-single-variable-calculus-fall-2006/)



### 第 2 步：通过这些课程和图书加深对深度学习的理解

[麻省理工学院课程 6.S191：深度学习简介](http://introtodeeplearning.com/)

Andrew Ng  [在 Coursera 上发布的《深度学习》专项课程](https://www.coursera.org/specializations/deep-learning)



⬆ 另请 ⬇ 阅读这些图书：

[《深度学习》（麻省理工学院出版社出版）](https://www.deeplearningbook.org/)

Michael Nielsen 编著的在线图书[神经网络与深度学习](http://neuralnetworksanddeeplearning.com/)



### 第 3 步：阅读论文并通过 TensorFlow 实现论文中的方法

[高级教程](https://www.tensorflow.org/tutorials?hl=zh-cn)

学习高级应用（[机器翻译](https://www.tensorflow.org/tutorials/text/transformer?hl=zh-cn)或[图片说明](https://www.tensorflow.org/tutorials/text/image_captioning?hl=zh-cn)）的最佳方式是阅读教程中链接到的论文。



# Appendix

## Mac M2 install Tensorflow GPU

https://developer.apple.com/metal/tensorflow-plugin/

For python >= 3.8 and tenforflow >= 2.13.

```bash
conda create -n tensorflow-gpu python=3.10

conda activate tensorflow-gpu
pip install tensorflow
pip install tensorflow-metal
```

```bash
conda list | grep tensorflow
# packages in environment at /opt/miniconda3/envs/tensorflow-gpu:
tensorflow                2.16.1                   pypi_0    pypi
tensorflow-io-gcs-filesystem 0.36.0                   pypi_0    pypi
tensorflow-metal          1.1.0                    pypi_0    pypi
```

Test using GPU.

```
You can also adjust the verbosity by changing the value of TF_CPP_MIN_LOG_LEVEL:

0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
```

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Setting log levels as above explained

import sys
import tensorflow as tf
import keras
import platform

print(f"Python               {sys.version}")
print(f"Python Platform:     {platform.platform()}")
print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version:       {keras.__version__}")
print("len(tf.config.list_physical_devices('GPU')) ", len(tf.config.list_physical_devices('GPU')))
print("tf.config.list_physical_devices('GPU')      ", tf.config.list_physical_devices('GPU'))

# Create some tensors
print("\nExample:")
tf.debugging.set_log_device_placement(True)
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)
print(f"{a} * {b} == {c}")
```

```python
import os
import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Setting log levels
cifar = keras.datasets.cifar100
(x_train, y_train), (x_test, y_test) = cifar.load_data()
model = keras.applications.ResNet50(
    include_top=True,
    weights=None,
    input_shape=(32, 32, 3),
    classes=100,)

loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
model.fit(x_train, y_train, epochs=5, batch_size=64)


# With Apple GPU used, one epoch took 536.0334 seconds
# With Apple CPU used, one epoch took 337.3022 seconds
```

Faced warnings.

```bash
Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.
```

