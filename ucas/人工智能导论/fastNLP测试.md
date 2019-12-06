
# 测试NLP项目

安装依赖


```python
! pip install fastNLP
```

    Requirement already satisfied: fastNLP in /home/ma-user/anaconda3/lib/python3.6/site-packages
    Requirement already satisfied: tqdm>=4.28.1 in /home/ma-user/anaconda3/lib/python3.6/site-packages (from fastNLP)
    Requirement already satisfied: numpy>=1.14.2 in /home/ma-user/anaconda3/lib/python3.6/site-packages (from fastNLP)
    Requirement already satisfied: torch>=1.0.0 in /home/ma-user/anaconda3/lib/python3.6/site-packages (from fastNLP)
    Requirement already satisfied: nltk>=3.4.1 in /home/ma-user/anaconda3/lib/python3.6/site-packages (from fastNLP)
    Requirement already satisfied: requests in /home/ma-user/anaconda3/lib/python3.6/site-packages (from fastNLP)
    Requirement already satisfied: spacy in /home/ma-user/anaconda3/lib/python3.6/site-packages (from fastNLP)
    Requirement already satisfied: prettytable>=0.7.2 in /home/ma-user/anaconda3/lib/python3.6/site-packages (from fastNLP)
    Requirement already satisfied: six in /home/ma-user/anaconda3/lib/python3.6/site-packages (from nltk>=3.4.1->fastNLP)
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /home/ma-user/anaconda3/lib/python3.6/site-packages (from requests->fastNLP)
    Requirement already satisfied: idna<2.7,>=2.5 in /home/ma-user/anaconda3/lib/python3.6/site-packages (from requests->fastNLP)
    Requirement already satisfied: urllib3<1.23,>=1.21.1 in /home/ma-user/anaconda3/lib/python3.6/site-packages (from requests->fastNLP)
    Requirement already satisfied: certifi>=2017.4.17 in /home/ma-user/anaconda3/lib/python3.6/site-packages (from requests->fastNLP)
    Requirement already satisfied: plac<1.2.0,>=0.9.6 in /home/ma-user/anaconda3/lib/python3.6/site-packages (from spacy->fastNLP)
    Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /home/ma-user/anaconda3/lib/python3.6/site-packages (from spacy->fastNLP)
    Requirement already satisfied: thinc<7.4.0,>=7.3.0 in /home/ma-user/anaconda3/lib/python3.6/site-packages (from spacy->fastNLP)
    Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /home/ma-user/anaconda3/lib/python3.6/site-packages (from spacy->fastNLP)
    Requirement already satisfied: setuptools in /home/ma-user/anaconda3/lib/python3.6/site-packages (from spacy->fastNLP)
    Requirement already satisfied: srsly<1.1.0,>=0.1.0 in /home/ma-user/anaconda3/lib/python3.6/site-packages (from spacy->fastNLP)
    Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /home/ma-user/anaconda3/lib/python3.6/site-packages (from spacy->fastNLP)
    Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /home/ma-user/anaconda3/lib/python3.6/site-packages (from spacy->fastNLP)
    Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /home/ma-user/anaconda3/lib/python3.6/site-packages (from spacy->fastNLP)
    Requirement already satisfied: blis<0.5.0,>=0.4.0 in /home/ma-user/anaconda3/lib/python3.6/site-packages (from spacy->fastNLP)
    Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /home/ma-user/anaconda3/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy->fastNLP)
    Requirement already satisfied: zipp>=0.5 in /home/ma-user/anaconda3/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy->fastNLP)
    Requirement already satisfied: more-itertools in /home/ma-user/anaconda3/lib/python3.6/site-packages (from zipp>=0.5->importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy->fastNLP)
    [33mYou are using pip version 9.0.1, however version 19.3.1 is available.
    You should consider upgrading via the 'pip install --upgrade pip' command.[0m


加载数据


```python
from fastNLP.io import ChnSentiCorpLoader

loader = ChnSentiCorpLoader()        # 初始化一个中文情感分类的loader
data_dir = loader.download()         # 这一行代码将自动下载数据到默认的缓存地址, 并将该地址返回
data_bundle = loader.load(data_dir)  # 这一行代码将从{data_dir}处读取数据至DataBundle
```


```python
print(data_bundle)
```

    In total 3 datasets:
    	dev has 1200 instances.
    	test has 1200 instances.
    	train has 9600 instances.
    



```python
print(data_bundle.get_dataset('train')[:2])  # 查看Train集前两个sample
```

    +-------------------------------------------+--------+
    | raw_chars                                 | target |
    +-------------------------------------------+--------+
    | 选择珠江花园的原因就是方便，有电动扶梯... | 1      |
    | 15.4寸笔记本的键盘确实爽，基本跟台式机... | 1      |
    +-------------------------------------------+--------+


数据预处理


```python
from fastNLP.io import ChnSentiCorpPipe

pipe = ChnSentiCorpPipe()
data_bundle = pipe.process(data_bundle)  # 所有的Pipe都实现了process()方法，且输入输出都为DataBundle类型

print(data_bundle)  # 打印data_bundle，查看其变化
```

    In total 3 datasets:
    	dev has 1200 instances.
    	test has 1200 instances.
    	train has 9600 instances.
    In total 2 vocabs:
    	chars has 4409 entries.
    	target has 2 entries.
    



```python
print(data_bundle.get_dataset('train')[:2])
```

    +-------------------------+--------+------------------------+---------+
    | raw_chars               | target | chars                  | seq_len |
    +-------------------------+--------+------------------------+---------+
    | 选择珠江花园的原因就... | 0      | [338, 464, 1400, 78... | 106     |
    | 15.4寸笔记本的键盘确... | 0      | [50, 133, 20, 135, ... | 56      |
    +-------------------------+--------+------------------------+---------+



```python
char_vocab = data_bundle.get_vocab('chars')
print(char_vocab)
```

    Vocabulary(['选', '择', '珠', '江', '花']...)



```python
index = char_vocab.to_index('选')
print("'选'的index是{}".format(index))  # 这个值与上面打印出来的第一个instance的chars的第一个index是一致的
print("index:{}对应的汉字是{}".format(index, char_vocab.to_word(index)))
```

    '选'的index是338
    index:338对应的汉字是选


选择预训练词向量


```python
from fastNLP.embeddings import StaticEmbedding

word2vec_embed = StaticEmbedding(char_vocab, model_dir_or_name='cn-char-fastnlp-100d')
```

    Found 4321 out of 4409 words in the pre-training embedding.


创建模型


```python
import torch
import torch.nn as nn

from fastNLP.core.const import Const as C
from fastNLP.core.utils import seq_len_to_mask
from fastNLP.embeddings import embedding
from fastNLP.modules import encoder

class CNNText(torch.nn.Module):
    """
    使用CNN进行文本分类的模型
    'Yoon Kim. 2014. Convolution Neural Networks for Sentence Classification.'
    
    """

    def __init__(self, embed,
                 num_classes,
                 kernel_nums=(30, 40, 50),
                 kernel_sizes=(1, 3, 5),
                 dropout=0.5):
        """
        
        :param tuple(int,int),torch.FloatTensor,nn.Embedding,numpy.ndarray embed: Embedding的大小(传入tuple(int, int),
            第一个int为vocab_zie, 第二个int为embed_dim); 如果为Tensor, Embedding, ndarray等则直接使用该值初始化Embedding
        :param int num_classes: 一共有多少类
        :param int,tuple(int) kernel_sizes: 输出channel的kernel大小。
        :param float dropout: Dropout的大小
        """
        super(CNNText, self).__init__()

        # no support for pre-trained embedding currently
        self.embed = embedding.Embedding(embed)
        self.conv_pool = encoder.ConvMaxpool(
            in_channels=self.embed.embedding_dim,
            out_channels=kernel_nums,
            kernel_sizes=kernel_sizes)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(sum(kernel_nums), num_classes)

    def forward(self, chars, seq_len=None):
        """

        :param torch.LongTensor words: [batch_size, seq_len]，句子中word的index
        :param torch.LongTensor seq_len:  [batch,] 每个句子的长度
        :return output: dict of torch.LongTensor, [batch_size, num_classes]
        """
        x = self.embed(chars)  # [N,L] -> [N,L,C]
        if seq_len is not None:
            mask = seq_len_to_mask(seq_len)
            x = self.conv_pool(x, mask)
        else:
            x = self.conv_pool(x)  # [N,L,C] -> [N,C]
        x = self.dropout(x)
        x = self.fc(x)  # [N,C] -> [N, N_class]
        return {C.OUTPUT: x}

    def predict(self, chars, seq_len=None):
        """
        :param torch.LongTensor words: [batch_size, seq_len]，句子中word的index
        :param torch.LongTensor seq_len:  [batch,] 每个句子的长度

        :return predict: dict of torch.LongTensor, [batch_size, ]
        """
        output = self(chars, seq_len)
        _, predict = output[C.OUTPUT].max(dim=1)
        return {C.OUTPUT: predict}

# 初始化模型
model = CNNText(word2vec_embed, len(data_bundle.get_vocab('target')))
```


```python
print(model)
```

    CNNText(
      (embed): Embedding(
        (embed): StaticEmbedding(
          (dropout_layer): Dropout(p=0, inplace=False)
          (embedding): Embedding(4385, 100, padding_idx=0)
        )
        (dropout): Dropout(p=0.0, inplace=False)
      )
      (conv_pool): ConvMaxpool(
        (convs): ModuleList(
          (0): Conv1d(100, 30, kernel_size=(1,), stride=(1,), bias=False)
          (1): Conv1d(100, 40, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
          (2): Conv1d(100, 50, kernel_size=(5,), stride=(1,), padding=(2,), bias=False)
        )
      )
      (dropout): Dropout(p=0.5, inplace=False)
      (fc): Linear(in_features=120, out_features=2, bias=True)
    )


训练


```python
from fastNLP import Trainer
from fastNLP import CrossEntropyLoss
from torch.optim import Adam
from fastNLP import AccuracyMetric
from fastNLP.core import EarlyStopCallback

loss = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)
metric = AccuracyMetric()
device = 0 if torch.cuda.is_available() else 'cpu'  # 如果有gpu的话在gpu上运行，训练速度会更快

trainer = Trainer(train_data=data_bundle.get_dataset('train'), model=model, loss=loss,
                  optimizer=optimizer, batch_size=32, dev_data=data_bundle.get_dataset('dev'),
                  metrics=metric, device=device, callbacks=[EarlyStopCallback(2)])
trainer.train()  # 开始训练，训练完成之后默认会加载在dev上表现最好的模型

# 在测试集上测试一下模型的性能
from fastNLP import Tester
print("Performance on test is:")
tester = Tester(data=data_bundle.get_dataset('test'), model=model, metrics=metric, batch_size=64, device=device)
tester.test()
```

    input fields after batch(if batch size is 2):
    	target: (1)type:torch.Tensor (2)dtype:torch.int64, (3)shape:torch.Size([2]) 
    	chars: (1)type:torch.Tensor (2)dtype:torch.int64, (3)shape:torch.Size([2, 106]) 
    	seq_len: (1)type:torch.Tensor (2)dtype:torch.int64, (3)shape:torch.Size([2]) 
    target fields after batch(if batch size is 2):
    	target: (1)type:torch.Tensor (2)dtype:torch.int64, (3)shape:torch.Size([2]) 
    
    training epochs started 2019-12-06-16-57-01



<p>Failed to display Jupyter Widget of type <code>HBox</code>.</p>
<p>
  If you're reading this message in the Jupyter Notebook or JupyterLab Notebook, it may mean
  that the widgets JavaScript is still loading. If this message persists, it
  likely means that the widgets JavaScript library is either not installed or
  not enabled. See the <a href="https://ipywidgets.readthedocs.io/en/stable/user_install.html">Jupyter
  Widgets Documentation</a> for setup instructions.
</p>
<p>
  If you're reading this message in another frontend (for example, a static
  rendering on GitHub or <a href="https://nbviewer.jupyter.org/">NBViewer</a>),
  it may mean that your frontend doesn't currently support widgets.
</p>




<p>Failed to display Jupyter Widget of type <code>HBox</code>.</p>
<p>
  If you're reading this message in the Jupyter Notebook or JupyterLab Notebook, it may mean
  that the widgets JavaScript is still loading. If this message persists, it
  likely means that the widgets JavaScript library is either not installed or
  not enabled. See the <a href="https://ipywidgets.readthedocs.io/en/stable/user_install.html">Jupyter
  Widgets Documentation</a> for setup instructions.
</p>
<p>
  If you're reading this message in another frontend (for example, a static
  rendering on GitHub or <a href="https://nbviewer.jupyter.org/">NBViewer</a>),
  it may mean that your frontend doesn't currently support widgets.
</p>



    Evaluate data in 0.71 seconds!
    Evaluation on dev at Epoch 1/10. Step:300/3000: 
    AccuracyMetric: acc=0.849167
    



<p>Failed to display Jupyter Widget of type <code>HBox</code>.</p>
<p>
  If you're reading this message in the Jupyter Notebook or JupyterLab Notebook, it may mean
  that the widgets JavaScript is still loading. If this message persists, it
  likely means that the widgets JavaScript library is either not installed or
  not enabled. See the <a href="https://ipywidgets.readthedocs.io/en/stable/user_install.html">Jupyter
  Widgets Documentation</a> for setup instructions.
</p>
<p>
  If you're reading this message in another frontend (for example, a static
  rendering on GitHub or <a href="https://nbviewer.jupyter.org/">NBViewer</a>),
  it may mean that your frontend doesn't currently support widgets.
</p>



    Evaluate data in 0.7 seconds!
    Evaluation on dev at Epoch 2/10. Step:600/3000: 
    AccuracyMetric: acc=0.890833
    



<p>Failed to display Jupyter Widget of type <code>HBox</code>.</p>
<p>
  If you're reading this message in the Jupyter Notebook or JupyterLab Notebook, it may mean
  that the widgets JavaScript is still loading. If this message persists, it
  likely means that the widgets JavaScript library is either not installed or
  not enabled. See the <a href="https://ipywidgets.readthedocs.io/en/stable/user_install.html">Jupyter
  Widgets Documentation</a> for setup instructions.
</p>
<p>
  If you're reading this message in another frontend (for example, a static
  rendering on GitHub or <a href="https://nbviewer.jupyter.org/">NBViewer</a>),
  it may mean that your frontend doesn't currently support widgets.
</p>



    Evaluate data in 0.75 seconds!
    Evaluation on dev at Epoch 3/10. Step:900/3000: 
    AccuracyMetric: acc=0.9025
    



<p>Failed to display Jupyter Widget of type <code>HBox</code>.</p>
<p>
  If you're reading this message in the Jupyter Notebook or JupyterLab Notebook, it may mean
  that the widgets JavaScript is still loading. If this message persists, it
  likely means that the widgets JavaScript library is either not installed or
  not enabled. See the <a href="https://ipywidgets.readthedocs.io/en/stable/user_install.html">Jupyter
  Widgets Documentation</a> for setup instructions.
</p>
<p>
  If you're reading this message in another frontend (for example, a static
  rendering on GitHub or <a href="https://nbviewer.jupyter.org/">NBViewer</a>),
  it may mean that your frontend doesn't currently support widgets.
</p>



    Evaluate data in 0.72 seconds!
    Evaluation on dev at Epoch 4/10. Step:1200/3000: 
    AccuracyMetric: acc=0.906667
    



<p>Failed to display Jupyter Widget of type <code>HBox</code>.</p>
<p>
  If you're reading this message in the Jupyter Notebook or JupyterLab Notebook, it may mean
  that the widgets JavaScript is still loading. If this message persists, it
  likely means that the widgets JavaScript library is either not installed or
  not enabled. See the <a href="https://ipywidgets.readthedocs.io/en/stable/user_install.html">Jupyter
  Widgets Documentation</a> for setup instructions.
</p>
<p>
  If you're reading this message in another frontend (for example, a static
  rendering on GitHub or <a href="https://nbviewer.jupyter.org/">NBViewer</a>),
  it may mean that your frontend doesn't currently support widgets.
</p>



    Evaluate data in 0.83 seconds!
    Evaluation on dev at Epoch 5/10. Step:1500/3000: 
    AccuracyMetric: acc=0.8925
    



<p>Failed to display Jupyter Widget of type <code>HBox</code>.</p>
<p>
  If you're reading this message in the Jupyter Notebook or JupyterLab Notebook, it may mean
  that the widgets JavaScript is still loading. If this message persists, it
  likely means that the widgets JavaScript library is either not installed or
  not enabled. See the <a href="https://ipywidgets.readthedocs.io/en/stable/user_install.html">Jupyter
  Widgets Documentation</a> for setup instructions.
</p>
<p>
  If you're reading this message in another frontend (for example, a static
  rendering on GitHub or <a href="https://nbviewer.jupyter.org/">NBViewer</a>),
  it may mean that your frontend doesn't currently support widgets.
</p>



    Evaluate data in 0.8 seconds!
    Evaluation on dev at Epoch 6/10. Step:1800/3000: 
    AccuracyMetric: acc=0.908333
    



<p>Failed to display Jupyter Widget of type <code>HBox</code>.</p>
<p>
  If you're reading this message in the Jupyter Notebook or JupyterLab Notebook, it may mean
  that the widgets JavaScript is still loading. If this message persists, it
  likely means that the widgets JavaScript library is either not installed or
  not enabled. See the <a href="https://ipywidgets.readthedocs.io/en/stable/user_install.html">Jupyter
  Widgets Documentation</a> for setup instructions.
</p>
<p>
  If you're reading this message in another frontend (for example, a static
  rendering on GitHub or <a href="https://nbviewer.jupyter.org/">NBViewer</a>),
  it may mean that your frontend doesn't currently support widgets.
</p>



    Evaluate data in 0.77 seconds!
    Evaluation on dev at Epoch 7/10. Step:2100/3000: 
    AccuracyMetric: acc=0.916667
    



<p>Failed to display Jupyter Widget of type <code>HBox</code>.</p>
<p>
  If you're reading this message in the Jupyter Notebook or JupyterLab Notebook, it may mean
  that the widgets JavaScript is still loading. If this message persists, it
  likely means that the widgets JavaScript library is either not installed or
  not enabled. See the <a href="https://ipywidgets.readthedocs.io/en/stable/user_install.html">Jupyter
  Widgets Documentation</a> for setup instructions.
</p>
<p>
  If you're reading this message in another frontend (for example, a static
  rendering on GitHub or <a href="https://nbviewer.jupyter.org/">NBViewer</a>),
  it may mean that your frontend doesn't currently support widgets.
</p>



    Evaluate data in 0.64 seconds!
    Evaluation on dev at Epoch 8/10. Step:2400/3000: 
    AccuracyMetric: acc=0.911667
    



<p>Failed to display Jupyter Widget of type <code>HBox</code>.</p>
<p>
  If you're reading this message in the Jupyter Notebook or JupyterLab Notebook, it may mean
  that the widgets JavaScript is still loading. If this message persists, it
  likely means that the widgets JavaScript library is either not installed or
  not enabled. See the <a href="https://ipywidgets.readthedocs.io/en/stable/user_install.html">Jupyter
  Widgets Documentation</a> for setup instructions.
</p>
<p>
  If you're reading this message in another frontend (for example, a static
  rendering on GitHub or <a href="https://nbviewer.jupyter.org/">NBViewer</a>),
  it may mean that your frontend doesn't currently support widgets.
</p>



    Evaluate data in 0.78 seconds!
    Evaluation on dev at Epoch 9/10. Step:2700/3000: 
    AccuracyMetric: acc=0.915
    



<p>Failed to display Jupyter Widget of type <code>HBox</code>.</p>
<p>
  If you're reading this message in the Jupyter Notebook or JupyterLab Notebook, it may mean
  that the widgets JavaScript is still loading. If this message persists, it
  likely means that the widgets JavaScript library is either not installed or
  not enabled. See the <a href="https://ipywidgets.readthedocs.io/en/stable/user_install.html">Jupyter
  Widgets Documentation</a> for setup instructions.
</p>
<p>
  If you're reading this message in another frontend (for example, a static
  rendering on GitHub or <a href="https://nbviewer.jupyter.org/">NBViewer</a>),
  it may mean that your frontend doesn't currently support widgets.
</p>



    Evaluate data in 0.74 seconds!
    Early Stopping triggered in epoch 10!
    
    In Epoch:7/Step:2100, got best dev performance:
    AccuracyMetric: acc=0.916667
    Reloaded the best model.
    Performance on test is:



<p>Failed to display Jupyter Widget of type <code>HBox</code>.</p>
<p>
  If you're reading this message in the Jupyter Notebook or JupyterLab Notebook, it may mean
  that the widgets JavaScript is still loading. If this message persists, it
  likely means that the widgets JavaScript library is either not installed or
  not enabled. See the <a href="https://ipywidgets.readthedocs.io/en/stable/user_install.html">Jupyter
  Widgets Documentation</a> for setup instructions.
</p>
<p>
  If you're reading this message in another frontend (for example, a static
  rendering on GitHub or <a href="https://nbviewer.jupyter.org/">NBViewer</a>),
  it may mean that your frontend doesn't currently support widgets.
</p>



    Evaluate data in 0.81 seconds!
    [tester] 
    AccuracyMetric: acc=0.919167





    {'AccuracyMetric': {'acc': 0.919167}}



预测


```python
from fastNLP.core.predictor import Predictor
from fastNLP import Const
from fastNLP import DataSet
```


```python
input_data = [list('买灯难，难于上青天。[悲伤]'),
             list('以后都卖切糕，保证发财。比金子都贵[衰]'),
             list('回复@盼盼snow:#周末去哪儿#现在圈来得及。[哈哈] //@盼盼snow:忘了圈@新鲜旅厦门 了，肿么办……[汗]'),
             list('粑粑麻麻年轻时的照片，哥太帅了，神马李敏镐之类的都边儿靠[哈哈][哈哈]')]
```


```python
dataset = DataSet({Const.CHAR_INPUT: input_data})
dataset.add_seq_len(field_name=Const.CHAR_INPUT)
```




    +------------------------------------------+---------+
    | chars                                    | seq_len |
    +------------------------------------------+---------+
    | ['买', '灯', '难', '，', '难', '于', ... | 14      |
    | ['以', '后', '都', '卖', '切', '糕', ... | 20      |
    | ['回', '复', '@', '盼', '盼', 's', 'n... | 59      |
    | ['粑', '粑', '麻', '麻', '年', '轻', ... | 36      |
    +------------------------------------------+---------+




```python
# 特征预处理
dataset.set_input(Const.CHAR_INPUT, Const.INPUT_LEN)
feature_vocabs = data_bundle.vocabs[Const.CHAR_INPUT]  # 提取特征词典
feature_vocabs.index_dataset(dataset, field_name=Const.CHAR_INPUT)
```




    Vocabulary(['选', '择', '珠', '江', '花']...)




```python
predictor = Predictor(model)
# 传入的dict的每个key的value应该为具有相同长度的list
batch_output = predictor.predict(data=dataset, seq_len_field_name=Const.INPUT_LEN)
print(batch_output)
pred_results = batch_output.get('pred')
print('pred results:{}'.format(pred_results))
```

    defaultdict(<class 'list'>, {'pred': [0, 0, 1, 1]})
    pred results:[0, 0, 1, 1]



```python

```
