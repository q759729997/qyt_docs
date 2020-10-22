# Ubuntu

## 常用命令

- 安装软件

~~~python
apt-get update  # 更新安装源
apt-get install vim  # 安装vim编辑器
~~~

## 国内源设置

- 备份/etc/apt/sources.list

~~~shell
cp /etc/apt/sources.list /etc/apt/sources.list.bak
~~~

- 在/etc/apt/sources.list文件前面添加如下条目

~~~shell
# 添加阿里源
deb http://mirrors.aliyun.com/ubuntu/ bionic main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ bionic-security main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ bionic-updates main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ bionic-proposed main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ bionic-backports main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ bionic main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ bionic-security main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ bionic-updates main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ bionic-proposed main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ bionic-backports main restricted universe multiverse
~~~

- 执行如下命令更新源

~~~shell
sudo apt-get update
sudo apt-get upgrade
~~~

## 设置中文语言环境

- 安装中文语言包

~~~python
apt-get install language-pack-zh-hans language-pack-zh-hans-base language-pack-gnome-zh-hans language-pack-gnome-zh-hans-base
apt-get install `check-language-support -l zh-hans`
locale-gen zh_CN.UTF-8
~~~

- 终端输入中文问题

~~~python
打开/etc/environment
在下面添加如下两行
LANG="zh_CN.UTF-8"
LANGUAGE="zh_CN:zh:en_US:en"

打开 /var/lib/locales/supported.d/local
添加zh_CN.GB2312字符集，如下：
en_US.UTF-8 UTF-8
zh_CN.UTF-8 UTF-8
zh_CN.GBK GBK
zh_CN GB2312
保存后，执行命令：
locale-gen

打开/etc/default/locale
修改为：
LANG="zh_CN.UTF-8"
LANGUAGE="zh_CN:zh:en_US:en"

vim ~/.bashrc (不要加 sudo)
複製下述這三行 貼在最後面
export LANG=LANG="zh_CN.utf-8"
export LANGUAGE="zh_CN:zh:en_US:en"
export LC_ALL="zh_CN.utf-8"

source ~/.bashrc

ls -al ~/ 查看是否有效
~~~

## docker镜像设置中文环境

- 拉取镜像

~~~shell
docker pull ubuntu:20.04
~~~

- 启动容器

~~~shell
docker run -it --name ubuntu_20_04 docker.io/ubuntu:20.04 /bin/bash
~~~

- 更新源

~~~shell
apt-get update
~~~

- 安装中文语言包

~~~shell
apt-get install language-pack-zh-hans language-pack-zh-hans-base language-pack-gnome-zh-hans language-pack-gnome-zh-hans-base

apt-get install `check-language-support -l zh-hans`

locale-gen zh_CN.UTF-8
~~~

- 终端输入中文问题

~~~python
# 安装vim
apt-get install vim
# 打开environment设置
vim /etc/environment
# 在下面添加如下两行
LANG="zh_CN.UTF-8"
LANGUAGE="zh_CN:zh:en_US:en"
# 打开 /var/lib/locales/supported.d/local
vim /var/lib/locales/supported.d/local
# 添加zh_CN.GB2312字符集，如下：
en_US.UTF-8 UTF-8
zh_CN.UTF-8 UTF-8
zh_CN.GBK GBK
zh_CN GB2312
# 保存后，执行命令：
locale-gen
# 打开/
vim /etc/default/locale
# 修改为：
LANG="zh_CN.UTF-8"
LANGUAGE="zh_CN:zh:en_US:en"

vim ~/.bashrc (不要加 sudo)
複製下述這三行 貼在最後面
export LANG=LANG="zh_CN.utf-8"
export LANGUAGE="zh_CN:zh:en_US:en"
export LC_ALL="zh_CN.utf-8"

source ~/.bashrc
# 查看是否有效
ls -al ~/ 
~~~

- 提交该镜像

~~~shell
docker commit ubuntu_20_04  ubuntu_zh:20_04
# 打包镜像
docker save -o ubuntu_zh_20_04.tar ubuntu_zh:20_04
# 上传
docker tag ubuntu_zh:20_04 docker.io/q759729997/ubuntu_zh:20_04
docker login
docker push docker.io/q759729997/ubuntu_zh:20_04
~~~

- 安装libreoffice

~~~shell
apt-get install libreoffice
apt-get install libreoffice-l10n-zh-cn libreoffice-help-zh-cn
# 提交镜像
docker commit ubuntu_20_04  ubuntu_zh_libreoffice:20_04
# 打包镜像
docker save -o ubuntu_zh_libreoffice_20_04.tar ubuntu_zh_libreoffice:20_04
~~~

- 测试libreoffice

~~~shell
# 拷贝测试文件
docker cp /root/temp/doc/11.doc ubuntu_20_04:/root/temp/
# doc 转 docx
soffice --invisible --headless --convert-to docx ./11.doc --outdir ./
soffice --invisible --headless --convert-to txt ./11.docx --outdir ./
soffice --invisible --headless --convert-to html ./11.docx --outdir ./
~~~
