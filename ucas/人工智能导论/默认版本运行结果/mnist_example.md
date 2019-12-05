
# 使用MoXing实现手写数字图像识别应用

  &#160;&#160;本内容主要介绍，如何使用MoXing实现手写数字图像的训练、测试应用。  
  
### [1. 准备数据](#data_prepare)
### [2. 训练模型](#train)
### [3. 预测](#predict)



## <a name="data_prepare">1. 准备数据</a>  
  &#160;&#160;从obs的mnist桶的mnist_data对象中下载MNIST数据集，并上传至私有的OBS桶中。
  
1.1 &#160; &#160; 下载MNIST数据集， 数据集文件说明如下：
- t10k-images-idx3-ubyte.gz：验证集，共包含10000个样本。
- t10k-labels-idx1-ubyte.gz：验证集标签，共包含10000个样本的类别标签。
- train-images-idx3-ubyte.gz：训练集，共包含60000个样本。
- train-labels-idx1-ubyte.gz：训练集标签，共包含60000个样本的类别标签。

1.2 &#160; &#160; .gz数据无需解压，分别上传至华为云OBS桶 ,该数据路径将设置为data_url。

# <a name="train">2. 训练模型</a>  

  &#160;&#160;通过import加载moxing的tensorflow模块 moxing.tensorflow 


```python
import moxing.tensorflow as mox
import os
```

    INFO:root:Using MoXing-v1.14.0-3c8d0e90
    INFO:root:Using OBS-Python-SDK-3.1.2
    INFO:tensorflow:Using TensorFlow-b'v1.13.1-0-g6612da8951'


根据数据存储和数据输出设置data_url和train_url


```python
####### your coding place： begin  ###########
# 此处必须修改为用户数据桶位置

#数据在OBS的存储位置。
# eg. s3:// ：统一路径输入
#     /uBucket ：桶名，用户的私有桶的名称 eg. bucket
#     /notebook/data/： 文件路径
# 参考网址：https://github.com/huaweicloud/ModelArts-Lab/blob/master/official_examples/docs/%E4%BD%BF%E7%94%A8Notebook%E5%AE%9E%E7%8E%B0%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB.md
data_url = 's3://qyt-mnist-data/dataset-mnist/' 

####### your coding place： end  ###########
```


```python
train_url = './cache/log/'          #训练输出位置。
if not mox.file.exists(data_url):
    raise ValueError('Plese verify your data url!')
if mox.file.exists(train_url):
    mox.file.remove(train_url,recursive=True)
mox.file.make_dirs(train_url)
```

 通过mox 能够将数据拷贝到本地，这样能够加快训练。操作如下：


```python
# 本地创建数据存储文件夹
local_url = './cache/local_data/'
if mox.file.exists(local_url):
    mox.file.remove(local_url,recursive=True)
os.makedirs(local_url)

#将私有桶中的数据拷贝到本地mox.file.copy_parallel（）
"""
  Copy all files in src_url to dst_url. Same usage as `shutil.copytree`.
  Note that this method can only copy a directory. If you want to copy a single file,
  please use `mox.file.copy`

  Example::

    copy_parallel(src_url='/tmp', dst_url='s3://bucket_name/my_data')

  Assuming files in `/tmp` are:

  * /tmp:
      * |- train
          * |- 1.jpg
          * |- 2.jpg
      * |- eval
          * |- 3.jpg
          * |- 4.jpg

  Then files after copy in `s3://bucket_name/my_data` are:

  * s3://bucket_name/my_data:
      * |- train
          * |- 1.jpg
          * |- 2.jpg
      * |- eval
          * |- 3.jpg
          * |- 4.jpg

  Directory `tmp` will not be copied. If `file_list` is `['train/1.jpg', 'eval/4.jpg']`,
  then files after copy in `s3://bucket_name/my_data` are:

  * s3://bucket_name/my_data
      * |- train
          * |- 1.jpg
      * |- eval
          * |- 4.jpg

  :param src_url: Source path or s3 url
  :param dst_url: Destination path or s3 url
  :param file_list: A list of relative path to `src_url` of files need to be copied.
  :param threads: Number of threads or processings in Pool.
  :param is_processing: If True, multiprocessing is used. If False, multithreading is used.
  :param use_queue: Whether use queue to manage downloading list.
  :return: None
"""
mox.file.copy_parallel(data_url, local_url)
data_url = local_url
os.listdir(data_url)
```




    ['t10k-labels-idx1-ubyte',
     't10k-images-idx3-ubyte.gz',
     't10k-labels-idx1-ubyte.gz',
     'Mnist-Data-Set.zip',
     't10k-images-idx3-ubyte',
     'test.jpg',
     'train-labels-idx1-ubyte',
     'train-labels-idx1-ubyte.gz',
     'train-images-idx3-ubyte.gz',
     'train-images-idx3-ubyte']




```python
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from __future__ import print_function
from __future__ import unicode_literals
```

**说明 1**  &#160; &#160; 函数 tf.flags.DEFINE_string('data_url', None, 'Dir of dataset')  数据路径。
                  函数tf.flags.DEFINE_string('train_url', None, 'Train Url') 日志以及生产模型的存储路径。 当脚本运行的时候可以利用tf.flags传入参数。


```python
tf.flags.DEFINE_string('data_url', None, 'Dir of dataset')
tf.flags.DEFINE_string('train_url', None, 'Train Url')

flags = tf.flags.FLAGS

filenames = ['train-images-idx3-ubyte.gz','train-labels-idx1-ubyte.gz','t10k-images-idx3-ubyte.gz',
             't10k-labels-idx1-ubyte.gz']

for filename in filenames:
  filepath = os.path.join(data_url, filename)
  if not mox.file.exists(filepath):
    raise ValueError('MNIST dataset file %s not found in %s' % (filepath, local_url))
```

####  &#160;&#160;训练的main函数包含三个部分，输入定义、模型定义和运行。

1） 输入函数：input_fn(run_mode, **kwargs) 用户可以根据自己的输入编写。本例中通过迭代的方式从数据集中取数据。


2） 模型定义：def model_fn(inputs, run_mode, **kwargs): 模型结构定义函数，返回 mox.ModelSpec(），用户作业模式定义返回值。
但需要满足如下条件：

 &#160;&#160; For run_mode == ModeKeys.TRAIN: `loss` is required.
  
  &#160;&#160;  For run_mode == ModeKeys.EVAL: `log_info` is required.
  
  &#160;&#160;  For run_mode == ModeKeys.PREDICT: `output_info` is required.
  
  &#160;&#160;  For run_mode == ModeKeys.EXPORT: `export_spec` is required.
  

3） 执行训练： mox.run(），训练的过程中可指定optimizer的一些设置，训练batch的大小等，设置内容如下：


 &#160;&#160; 输入函数， input_fn: An input_fn defined by user. Allows tfrecord or python data. Returns  input tensor list.
 
 &#160;&#160;  模型函数， model_fn: A model_fn defined by user. Returns `mox.ModelSpec`.
  
  &#160;&#160; optimizer定义， optimizer_fn: An optimizer_fn defined by user. Returns an optimizer.
  
  &#160;&#160; 运行模式选择， run_mode: Only takes mox.ModeKeys.TRAIN or mox.ModeKeys.EVAL or mox.ModeKeys.PREDICT
  
  &#160;&#160; batch大小设置， batch_size: Mini-batch size.
  
 &#160;&#160;  是否自动化batch， auto_batch: If True, an extra dimension of batch_size will be expanded to the first
                     dimension of the return value from `get_split`. Default to True.
                     
  &#160;&#160; 日志以及checkpoint保存位置， log_dir: The directory to save summaries and checkpoints.
  
  &#160;&#160; 最大数量，  max_number_of_steps: Maximum steps for each worker.
                          
  &#160;&#160; 日志打印， log_every_n_steps: Step period to print logs to std I/O.
     
  &#160;&#160; 是否输出模型， export_model: True or False. Where to export model after running the job.




```python
def main(*args):
  flags.data_url = data_url
  flags.train_url = train_url
  mnist = input_data.read_data_sets(flags.data_url, one_hot=True)
        

  # define the input dataset, return image and label
  def input_fn(run_mode, **kwargs):
    def gen():
      while True:
        yield mnist.train.next_batch(50)
    ds = tf.data.Dataset.from_generator(
        gen, output_types=(tf.float32, tf.int64),
        output_shapes=(tf.TensorShape([None, 784]), tf.TensorShape([None, 10])))
    return ds.make_one_shot_iterator().get_next()


  # define the model for training or evaling.
  def model_fn(inputs, run_mode, **kwargs):
    x, y_ = inputs
    W = tf.get_variable(name='W', initializer=tf.zeros([784, 10]))
    b = tf.get_variable(name='b', initializer=tf.zeros([10]))
    y = tf.matmul(x, W) + b
    cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    predictions = tf.argmax(y, 1)
    correct_predictions = tf.equal(predictions, tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    export_spec = mox.ExportSpec(inputs_dict={'images': x}, outputs_dict={'predictions': predictions}, version='model')
    return mox.ModelSpec(loss=cross_entropy, log_info={'loss': cross_entropy, 'accuracy': accuracy},
                         export_spec=export_spec)


  mox.run(input_fn=input_fn,
          model_fn=model_fn,
          optimizer_fn=mox.get_optimizer_fn('sgd', learning_rate=0.01),
          run_mode=mox.ModeKeys.TRAIN,
          batch_size=32,  # 50
          auto_batch=False,
          log_dir=flags.train_url,
          max_number_of_steps=1000,
          log_every_n_steps=10,
          export_model=mox.ExportKeys.TF_SERVING)

if __name__ == '__main__':
  tf.app.run(main=main)
```

    WARNING:tensorflow:From <ipython-input-7-7fd406f6124a>:4: read_data_sets (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use alternatives such as official/mnist/dataset.py from tensorflow/models.
    WARNING:tensorflow:From /home/ma-user/anaconda3/envs/TensorFlow-1.13.1/lib/python3.6/site-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:260: maybe_download (from tensorflow.contrib.learn.python.learn.datasets.base) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please write your own downloading logic.
    WARNING:tensorflow:From /home/ma-user/anaconda3/envs/TensorFlow-1.13.1/lib/python3.6/site-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:262: extract_images (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use tf.data to implement this functionality.


    Extracting ./cache/local_data/train-images-idx3-ubyte.gz


    WARNING:tensorflow:From /home/ma-user/anaconda3/envs/TensorFlow-1.13.1/lib/python3.6/site-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:267: extract_labels (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use tf.data to implement this functionality.
    WARNING:tensorflow:From /home/ma-user/anaconda3/envs/TensorFlow-1.13.1/lib/python3.6/site-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:110: dense_to_one_hot (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use tf.one_hot on tensors.
    WARNING:tensorflow:From /home/ma-user/anaconda3/envs/TensorFlow-1.13.1/lib/python3.6/site-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:290: DataSet.__init__ (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use alternatives such as official/mnist/dataset.py from tensorflow/models.


    Extracting ./cache/local_data/train-labels-idx1-ubyte.gz
    Extracting ./cache/local_data/t10k-images-idx3-ubyte.gz
    Extracting ./cache/local_data/t10k-labels-idx1-ubyte.gz


    WARNING:tensorflow:From /home/ma-user/anaconda3/envs/TensorFlow-1.13.1/lib/python3.6/site-packages/tensorflow/python/framework/op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Colocations handled automatically by placer.
    WARNING:tensorflow:From /home/ma-user/anaconda3/envs/TensorFlow-1.13.1/lib/python3.6/site-packages/tensorflow/python/data/ops/dataset_ops.py:429: py_func (from tensorflow.python.ops.script_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    tf.py_func is deprecated in TF V2. Instead, use
        tf.py_function, which takes a python function which manipulates tf eager
        tensors instead of numpy arrays. It's easy to convert a tf eager tensor to
        an ndarray (just call tensor.numpy()) but having access to eager tensors
        means `tf.py_function`s can use accelerators such as GPUs as well as
        being differentiable using a gradient tape.
        
    WARNING:tensorflow:From /home/ma-user/anaconda3/envs/TensorFlow-1.13.1/lib/python3.6/site-packages/tensorflow/contrib/slim/python/slim/data/prefetch_queue.py:88: QueueRunner.__init__ (from tensorflow.python.training.queue_runner_impl) is deprecated and will be removed in a future version.
    Instructions for updating:
    To construct input pipelines, use the `tf.data` module.
    WARNING:tensorflow:From /home/ma-user/anaconda3/envs/TensorFlow-1.13.1/lib/python3.6/site-packages/tensorflow/contrib/slim/python/slim/data/prefetch_queue.py:88: add_queue_runner (from tensorflow.python.training.queue_runner_impl) is deprecated and will be removed in a future version.
    Instructions for updating:
    To construct input pipelines, use the `tf.data` module.
    WARNING:tensorflow:From /home/ma-user/anaconda3/envs/TensorFlow-1.13.1/lib/python3.6/site-packages/tensorflow/contrib/slim/python/slim/data/prefetch_queue.py:90: to_float (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.cast instead.
    WARNING:tensorflow:From <ipython-input-7-7fd406f6124a>:25: softmax_cross_entropy_with_logits (from tensorflow.python.ops.nn_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    
    Future major versions of TensorFlow will allow gradients to flow
    into the labels input on backprop by default.
    
    See `tf.nn.softmax_cross_entropy_with_logits_v2`.
    
    INFO:tensorflow:Create CheckpointSaverHook.
    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.
    WARNING:tensorflow:From /home/ma-user/anaconda3/envs/TensorFlow-1.13.1/lib/python3.6/site-packages/tensorflow/python/training/monitored_session.py:809: start_queue_runners (from tensorflow.python.training.queue_runner_impl) is deprecated and will be removed in a future version.
    Instructions for updating:
    To construct input pipelines, use the `tf.data` module.
    INFO:tensorflow:Saving checkpoints for 0 into ./cache/log/model.ckpt.
    INFO:tensorflow:Running will end at step: 50
    INFO:tensorflow:step: 0(global step: 0)	sample/sec: 97.193	loss: 2.303	accuracy: 0.140
    INFO:tensorflow:step: 10(global step: 10)	sample/sec: 21331.489	loss: 2.196	accuracy: 0.480
    INFO:tensorflow:step: 20(global step: 20)	sample/sec: 24905.869	loss: 2.113	accuracy: 0.480
    INFO:tensorflow:step: 30(global step: 30)	sample/sec: 27851.780	loss: 2.026	accuracy: 0.620
    INFO:tensorflow:step: 40(global step: 40)	sample/sec: 25362.382	loss: 1.950	accuracy: 0.760
    INFO:tensorflow:Sync to send FPS to non-chief workers.
    INFO:tensorflow:Saving checkpoints for 50 into ./cache/log/model.ckpt.
    WARNING:tensorflow:From <ipython-input-7-7fd406f6124a>:43: checkpoint_exists (from tensorflow.python.training.checkpoint_management) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use standard file APIs to check for files with this prefix.
    INFO:tensorflow:Create CheckpointSaverHook.
    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Restoring parameters from ./cache/log/model.ckpt-50
    WARNING:tensorflow:From /home/ma-user/anaconda3/envs/TensorFlow-1.13.1/lib/python3.6/site-packages/tensorflow/python/training/saver.py:1070: get_checkpoint_mtimes (from tensorflow.python.training.checkpoint_management) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use standard file utilities to get mtimes.
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.
    INFO:tensorflow:Saving checkpoints for 50 into ./cache/log/model.ckpt.
    INFO:tensorflow:Running will end at step: 100
    INFO:tensorflow:step: 50(global step: 50)	sample/sec: 493.925	loss: 1.890	accuracy: 0.720
    INFO:tensorflow:step: 60(global step: 60)	sample/sec: 29733.657	loss: 1.839	accuracy: 0.740
    INFO:tensorflow:step: 70(global step: 70)	sample/sec: 30407.279	loss: 1.649	accuracy: 0.780
    INFO:tensorflow:step: 80(global step: 80)	sample/sec: 28920.002	loss: 1.599	accuracy: 0.800
    INFO:tensorflow:step: 90(global step: 90)	sample/sec: 31782.555	loss: 1.561	accuracy: 0.780
    INFO:tensorflow:Sync to send FPS to non-chief workers.
    INFO:tensorflow:Saving checkpoints for 100 into ./cache/log/model.ckpt.
    INFO:tensorflow:Create CheckpointSaverHook.
    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Restoring parameters from ./cache/log/model.ckpt-100
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.
    INFO:tensorflow:Saving checkpoints for 100 into ./cache/log/model.ckpt.
    INFO:tensorflow:Running will end at step: 150
    INFO:tensorflow:step: 100(global step: 100)	sample/sec: 479.615	loss: 1.490	accuracy: 0.900
    INFO:tensorflow:step: 110(global step: 110)	sample/sec: 29001.238	loss: 1.515	accuracy: 0.800
    INFO:tensorflow:step: 120(global step: 120)	sample/sec: 32404.087	loss: 1.485	accuracy: 0.740
    INFO:tensorflow:step: 130(global step: 130)	sample/sec: 31004.326	loss: 1.380	accuracy: 0.800
    INFO:tensorflow:step: 140(global step: 140)	sample/sec: 31068.919	loss: 1.410	accuracy: 0.640
    INFO:tensorflow:Sync to send FPS to non-chief workers.
    INFO:tensorflow:Saving checkpoints for 150 into ./cache/log/model.ckpt.
    INFO:tensorflow:Create CheckpointSaverHook.
    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Restoring parameters from ./cache/log/model.ckpt-150
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.
    INFO:tensorflow:Saving checkpoints for 150 into ./cache/log/model.ckpt.
    INFO:tensorflow:Running will end at step: 200
    INFO:tensorflow:step: 150(global step: 150)	sample/sec: 472.104	loss: 1.370	accuracy: 0.820
    INFO:tensorflow:step: 160(global step: 160)	sample/sec: 28292.101	loss: 1.317	accuracy: 0.800
    INFO:tensorflow:step: 170(global step: 170)	sample/sec: 29146.086	loss: 1.325	accuracy: 0.840
    INFO:tensorflow:step: 180(global step: 180)	sample/sec: 30940.002	loss: 1.255	accuracy: 0.780
    INFO:tensorflow:step: 190(global step: 190)	sample/sec: 30897.267	loss: 1.202	accuracy: 0.880
    INFO:tensorflow:Sync to send FPS to non-chief workers.
    INFO:tensorflow:Saving checkpoints for 200 into ./cache/log/model.ckpt.
    INFO:tensorflow:Create CheckpointSaverHook.
    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Restoring parameters from ./cache/log/model.ckpt-200
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.
    INFO:tensorflow:Saving checkpoints for 200 into ./cache/log/model.ckpt.
    INFO:tensorflow:Running will end at step: 1000
    INFO:tensorflow:step: 200(global step: 200)	sample/sec: 481.138	loss: 1.191	accuracy: 0.820
    INFO:tensorflow:step: 210(global step: 210)	sample/sec: 29979.390	loss: 1.063	accuracy: 0.820
    INFO:tensorflow:step: 220(global step: 220)	sample/sec: 29550.358	loss: 1.166	accuracy: 0.760
    INFO:tensorflow:step: 230(global step: 230)	sample/sec: 27330.020	loss: 1.065	accuracy: 0.880
    INFO:tensorflow:step: 240(global step: 240)	sample/sec: 29779.838	loss: 1.173	accuracy: 0.740
    INFO:tensorflow:step: 250(global step: 250)	sample/sec: 29395.035	loss: 1.132	accuracy: 0.820
    INFO:tensorflow:step: 260(global step: 260)	sample/sec: 30215.607	loss: 0.942	accuracy: 0.840
    INFO:tensorflow:step: 270(global step: 270)	sample/sec: 30925.744	loss: 1.057	accuracy: 0.780
    INFO:tensorflow:step: 280(global step: 280)	sample/sec: 30762.716	loss: 0.994	accuracy: 0.760
    INFO:tensorflow:step: 290(global step: 290)	sample/sec: 26646.363	loss: 1.047	accuracy: 0.800
    INFO:tensorflow:global_step/sec: 571.232
    INFO:tensorflow:step: 300(global step: 300)	sample/sec: 13177.980	loss: 0.908	accuracy: 0.860
    INFO:tensorflow:step: 310(global step: 310)	sample/sec: 32256.123	loss: 0.945	accuracy: 0.780
    INFO:tensorflow:step: 320(global step: 320)	sample/sec: 28679.002	loss: 0.992	accuracy: 0.820
    INFO:tensorflow:step: 330(global step: 330)	sample/sec: 25243.131	loss: 0.988	accuracy: 0.780
    INFO:tensorflow:step: 340(global step: 340)	sample/sec: 27380.197	loss: 0.902	accuracy: 0.880
    INFO:tensorflow:step: 350(global step: 350)	sample/sec: 24389.920	loss: 0.963	accuracy: 0.760
    INFO:tensorflow:step: 360(global step: 360)	sample/sec: 32372.824	loss: 0.966	accuracy: 0.820
    INFO:tensorflow:step: 370(global step: 370)	sample/sec: 29427.259	loss: 0.703	accuracy: 0.960
    INFO:tensorflow:step: 380(global step: 380)	sample/sec: 33042.277	loss: 0.928	accuracy: 0.800
    INFO:tensorflow:step: 390(global step: 390)	sample/sec: 29979.390	loss: 0.763	accuracy: 0.940
    INFO:tensorflow:global_step/sec: 819.294
    INFO:tensorflow:step: 400(global step: 400)	sample/sec: 12270.774	loss: 0.883	accuracy: 0.800
    INFO:tensorflow:step: 410(global step: 410)	sample/sec: 27753.873	loss: 0.832	accuracy: 0.840
    INFO:tensorflow:step: 420(global step: 420)	sample/sec: 32094.148	loss: 1.074	accuracy: 0.780
    INFO:tensorflow:step: 430(global step: 430)	sample/sec: 25850.872	loss: 1.062	accuracy: 0.780
    INFO:tensorflow:step: 440(global step: 440)	sample/sec: 26572.506	loss: 0.656	accuracy: 0.920
    INFO:tensorflow:step: 450(global step: 450)	sample/sec: 27811.382	loss: 0.897	accuracy: 0.800
    INFO:tensorflow:step: 460(global step: 460)	sample/sec: 27098.269	loss: 0.825	accuracy: 0.820
    INFO:tensorflow:step: 470(global step: 470)	sample/sec: 29766.629	loss: 0.832	accuracy: 0.840
    INFO:tensorflow:step: 480(global step: 480)	sample/sec: 27645.258	loss: 0.740	accuracy: 0.880
    INFO:tensorflow:step: 490(global step: 490)	sample/sec: 32577.118	loss: 0.860	accuracy: 0.840
    INFO:tensorflow:global_step/sec: 812.066
    INFO:tensorflow:step: 500(global step: 500)	sample/sec: 12546.058	loss: 0.827	accuracy: 0.780
    INFO:tensorflow:step: 510(global step: 510)	sample/sec: 30441.762	loss: 0.737	accuracy: 0.820
    INFO:tensorflow:step: 520(global step: 520)	sample/sec: 29979.390	loss: 0.856	accuracy: 0.780
    INFO:tensorflow:step: 530(global step: 530)	sample/sec: 30421.063	loss: 0.876	accuracy: 0.820
    INFO:tensorflow:step: 540(global step: 540)	sample/sec: 32380.634	loss: 0.945	accuracy: 0.760
    INFO:tensorflow:step: 550(global step: 550)	sample/sec: 31521.308	loss: 0.733	accuracy: 0.840
    INFO:tensorflow:step: 560(global step: 560)	sample/sec: 25209.941	loss: 0.700	accuracy: 0.920
    INFO:tensorflow:step: 570(global step: 570)	sample/sec: 30497.098	loss: 0.750	accuracy: 0.840
    INFO:tensorflow:step: 580(global step: 580)	sample/sec: 26641.073	loss: 0.955	accuracy: 0.800
    INFO:tensorflow:step: 590(global step: 590)	sample/sec: 28429.936	loss: 0.636	accuracy: 0.960
    INFO:tensorflow:global_step/sec: 846.576
    INFO:tensorflow:step: 600(global step: 600)	sample/sec: 12967.896	loss: 0.687	accuracy: 0.880
    INFO:tensorflow:step: 610(global step: 610)	sample/sec: 29158.750	loss: 0.847	accuracy: 0.800
    INFO:tensorflow:step: 620(global step: 620)	sample/sec: 31425.364	loss: 0.835	accuracy: 0.800
    INFO:tensorflow:step: 630(global step: 630)	sample/sec: 30776.824	loss: 0.792	accuracy: 0.840
    INFO:tensorflow:step: 640(global step: 640)	sample/sec: 31184.416	loss: 0.638	accuracy: 0.920
    INFO:tensorflow:step: 650(global step: 650)	sample/sec: 26694.059	loss: 0.763	accuracy: 0.880
    INFO:tensorflow:step: 660(global step: 660)	sample/sec: 30833.386	loss: 0.711	accuracy: 0.840
    INFO:tensorflow:step: 670(global step: 670)	sample/sec: 29305.181	loss: 0.667	accuracy: 0.900
    INFO:tensorflow:step: 680(global step: 680)	sample/sec: 31395.960	loss: 0.919	accuracy: 0.800
    INFO:tensorflow:step: 690(global step: 690)	sample/sec: 29343.622	loss: 0.746	accuracy: 0.780
    INFO:tensorflow:global_step/sec: 831.266
    INFO:tensorflow:step: 700(global step: 700)	sample/sec: 12210.492	loss: 0.586	accuracy: 0.920
    INFO:tensorflow:step: 710(global step: 710)	sample/sec: 32217.410	loss: 0.745	accuracy: 0.840
    INFO:tensorflow:step: 720(global step: 720)	sample/sec: 30359.133	loss: 0.618	accuracy: 0.920
    INFO:tensorflow:step: 730(global step: 730)	sample/sec: 29317.983	loss: 0.675	accuracy: 0.860
    INFO:tensorflow:step: 740(global step: 740)	sample/sec: 26625.219	loss: 0.622	accuracy: 0.920
    INFO:tensorflow:step: 750(global step: 750)	sample/sec: 27257.865	loss: 0.746	accuracy: 0.800
    INFO:tensorflow:step: 760(global step: 760)	sample/sec: 30304.296	loss: 0.733	accuracy: 0.860
    INFO:tensorflow:step: 770(global step: 770)	sample/sec: 29819.535	loss: 0.613	accuracy: 0.900
    INFO:tensorflow:step: 780(global step: 780)	sample/sec: 27186.090	loss: 0.798	accuracy: 0.840
    INFO:tensorflow:step: 790(global step: 790)	sample/sec: 28994.973	loss: 0.536	accuracy: 0.880
    INFO:tensorflow:global_step/sec: 839.475
    INFO:tensorflow:step: 800(global step: 800)	sample/sec: 11854.595	loss: 0.729	accuracy: 0.880
    INFO:tensorflow:step: 810(global step: 810)	sample/sec: 29478.965	loss: 0.584	accuracy: 0.860
    INFO:tensorflow:step: 820(global step: 820)	sample/sec: 30805.079	loss: 0.786	accuracy: 0.800
    INFO:tensorflow:step: 830(global step: 830)	sample/sec: 28197.002	loss: 0.647	accuracy: 0.800
    INFO:tensorflow:step: 840(global step: 840)	sample/sec: 30379.748	loss: 0.574	accuracy: 0.940
    INFO:tensorflow:step: 850(global step: 850)	sample/sec: 31521.308	loss: 0.650	accuracy: 0.900
    INFO:tensorflow:step: 860(global step: 860)	sample/sec: 30019.622	loss: 0.632	accuracy: 0.860
    INFO:tensorflow:step: 870(global step: 870)	sample/sec: 28907.544	loss: 0.699	accuracy: 0.880
    INFO:tensorflow:step: 880(global step: 880)	sample/sec: 32720.070	loss: 0.670	accuracy: 0.840
    INFO:tensorflow:step: 890(global step: 890)	sample/sec: 27274.482	loss: 0.535	accuracy: 0.920
    INFO:tensorflow:global_step/sec: 833.999
    INFO:tensorflow:step: 900(global step: 900)	sample/sec: 13180.568	loss: 0.633	accuracy: 0.900
    INFO:tensorflow:step: 910(global step: 910)	sample/sec: 31286.184	loss: 0.733	accuracy: 0.800
    INFO:tensorflow:step: 920(global step: 920)	sample/sec: 26784.619	loss: 0.748	accuracy: 0.840
    INFO:tensorflow:step: 930(global step: 930)	sample/sec: 28460.078	loss: 0.504	accuracy: 0.940
    INFO:tensorflow:step: 940(global step: 940)	sample/sec: 31447.453	loss: 0.501	accuracy: 0.900
    INFO:tensorflow:step: 950(global step: 950)	sample/sec: 29681.054	loss: 0.425	accuracy: 0.940
    INFO:tensorflow:step: 960(global step: 960)	sample/sec: 21543.777	loss: 0.537	accuracy: 0.900
    INFO:tensorflow:step: 970(global step: 970)	sample/sec: 27497.998	loss: 0.797	accuracy: 0.760
    INFO:tensorflow:step: 980(global step: 980)	sample/sec: 31213.425	loss: 0.607	accuracy: 0.840
    INFO:tensorflow:step: 990(global step: 990)	sample/sec: 30427.959	loss: 0.693	accuracy: 0.820
    INFO:tensorflow:Saving checkpoints for 1000 into ./cache/log/model.ckpt.
    WARNING:tensorflow:From /home/ma-user/anaconda3/envs/TensorFlow-1.13.1/lib/python3.6/site-packages/tensorflow/python/training/saver.py:966: remove_checkpoint (from tensorflow.python.training.checkpoint_management) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use standard file APIs to delete files with this prefix.
    INFO:tensorflow:Ignoring --checkpoint_path because a checkpoint already exists in ./cache/log/
    WARNING:tensorflow:From /home/ma-user/anaconda3/envs/TensorFlow-1.13.1/lib/python3.6/site-packages/tensorflow/python/saved_model/signature_def_utils_impl.py:205: build_tensor_info (from tensorflow.python.saved_model.utils_impl) is deprecated and will be removed in a future version.
    Instructions for updating:
    This function will only be available through the v1 compatibility library as tf.compat.v1.saved_model.utils.build_tensor_info or tf.compat.v1.saved_model.build_tensor_info.
    INFO:tensorflow:No assets to save.
    INFO:tensorflow:No assets to write.
    INFO:tensorflow:Restoring parameters from ./cache/log/model.ckpt-1000
    INFO:tensorflow:SavedModel written to: ./cache/log/model/saved_model.pb



    An exception has occurred, use %tb to see the full traceback.


    SystemExit



    /home/ma-user/anaconda3/envs/TensorFlow-1.13.1/lib/python3.6/site-packages/IPython/core/interactiveshell.py:2918: UserWarning: To exit: use 'exit', 'quit', or Ctrl-D.
      warn("To exit: use 'exit', 'quit', or Ctrl-D.", stacklevel=1)


## <a name="predict">2. 预测</a>  



&#8195;&#8195; 在上面训练的基础上，我们可以直接用训练的模型进行预测作业。如读取OBS桶中的数字图片进行识别。input_fn 对输入图片进行简单处理，得到网络允许的输入tensor；model_fn定义一个预测内容，同时，还需定义一个对输出处理的函数output_fn，我们在改函数里对输出进行一个打印输出。
 
  还需在 mox.run()函数中加入如下参数：
  
 &#8195;&#8195; 输出函数 output_fn: A callback with args of results from sess.run.
   
&#8195;&#8195; 模型加载位置 checkpoint_path: Directory or file path of ckpt to restore when `run_mode` is 'evaluation'.
                          Useless when `run_mode` is 'train'.


```python
####### your coding place： begin###########

#此处必须修改为用户数据存储的OBS位置

# 预测图片在OBS的存储位置。
# eg. 图片名称：  image_number.jpg
#     存储位置为：bucket/test/
src_path = 's3://qyt-mnist-data/train-log/test.jpg'

####### your coding place： end  ###########
```


```python
# 可以利用moxing 将需要预测的图片从OBS拷贝到本地
if not mox.file.exists(src_path):
    raise ValueError('Plese verify your src_path!')
dst_path =  './cache/test.jpg'
mox.file.copy(src_path,dst_path)
```


```python
image_path = './cache/test.jpg'            # 指定图片位置
checkpoint_url = './cache/log/'         # 指定checkpoint位置，即上一步训练指定的路径的位置。
print(mox.file.exists(image_path))
```

    True



```python
import moxing.tensorflow as mox
import os
import tensorflow as tf
from __future__ import print_function
from __future__ import unicode_literals

```


```python
def predict(*args):
  def input_fn(run_mode, **kwargs):
    image = tf.read_file(image_path)
    img = tf.image.decode_jpeg(image, channels=1)
    img = tf.image.resize_images(img, [28, 28], 0)
    img = tf.reshape(img, [784])
    return img

  def model_fn(inputs, run_mode, **kwargs):
    x = inputs
    W1 = tf.get_variable(name='W', initializer=tf.zeros([784, 10]))
    b1 = tf.get_variable(name='b', initializer=tf.zeros([10]))
    y = tf.matmul(x, W1) + b1
    predictions = tf.argmax(y, 1)
    return mox.ModelSpec(output_info={'predict': predictions})

  def output_fn(outputs):
    for output in outputs:
      result = output['predict']
      print("The result：",result)

  mox.run(input_fn=input_fn,
          model_fn=model_fn,
          output_fn=output_fn,
          run_mode=mox.ModeKeys.PREDICT,
          batch_size=1,
          auto_batch=False,
          max_number_of_steps=1,
          output_every_n_steps=1,
          checkpoint_path=checkpoint_url)
if __name__ == '__main__':
  tf.app.run(main=predict)
```

    INFO:tensorflow:Graph was finalized.
    INFO:tensorflow:Restoring parameters from ./cache/log/model.ckpt-1000
    INFO:tensorflow:Running local_init_op.
    INFO:tensorflow:Done running local_init_op.
    INFO:tensorflow:	[1 examples]


    The result： [7]



    An exception has occurred, use %tb to see the full traceback.


    SystemExit



    /home/ma-user/anaconda3/envs/TensorFlow-1.13.1/lib/python3.6/site-packages/IPython/core/interactiveshell.py:2918: UserWarning: To exit: use 'exit', 'quit', or Ctrl-D.
      warn("To exit: use 'exit', 'quit', or Ctrl-D.", stacklevel=1)


通过预测，我们能够看到结果输出。

