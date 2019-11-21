# conda简易教程

## 简介

-  Conda 是一个开源的软件包管理系统和环境管理系统，用于安装多个版本的软件包及其依赖关系，并在它们之间轻松切换。 （来源于百度百科）

## 使用

- 进入与离开环境

~~~python
source activate python36 或  conda activate python36 # 进入环境，Linux需加source
source deactivate 或  conda deactivate # 退出环境，Linux需加source
conda info  # 查看基本环境信息，包括Python版本号，运行环境等等
conda env list  # 查看所有环境
~~~

- 安装与卸载环境

~~~python
conda create --name python36 python=3.6  # 创建环境，并指明python版本
conda env remove -n python36 --all  # 删除环境，python36为环境名称，--all为删除该环境下所有包
conda create -n myenv --clone /data/conda_env  # 导入环境
/miniconda3/envs/python36/lib/python3.6/site-packages # 包位置
~~~

## 安装

- Linux下的安装

~~~python
mkdir software  # 具体路径根据自己实际情况进行选择，请在自己目录下面进行相关操作。
cd software
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  # 若需要新的安装包，自行在下面网站进行查找：https://repo.continuum.io/miniconda
安装 bzip2  # 例如 apt-get install bzip2
sh Miniconda3-latest-Linux-x86_64.sh  # 安装conda
source ~/.bashrc  # 安装完毕后,使环境变量生效
conda env list  # 执行此命令检查是否安装成功
~~~

- Windows下的安装：可在下面网页搜索对应Windows的安装包<https://repo.continuum.io/miniconda/>

## 加速下载

- 设置pip源

~~~python
mkdir ~/.pip
cd ~/.pip/
vim pip.conf
# 以下为设置内容
[global]
index-url=http://mirrors.aliyun.com/pypi/simple/
[install]
trusted-host=http://mirrors.aliyun.com/pypi/simple/
~~~

- pip源列表

~~~python
清华大学：https://pypi.tuna.tsinghua.edu.cn/simple
阿里云：http://mirrors.aliyun.com/pypi/simple/
豆瓣：http://pypi.douban.com/simple/
~~~