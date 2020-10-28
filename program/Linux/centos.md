# centos

- CentOS 国内镜像下载地址 ：

~~~wiki
http://mirrors.aliyun.com/centos/7/isos/x86_64/ 阿里云开源镜像
https://mirrors.cnnic.cn/centos/7/isos/x86_64/ 清华大学开源镜像

http://centos.ustc.edu.cn/centos/7/isos/x86_64/ 中国科学技术大学
http://ftp.sjtu.edu.cn/centos/7/isos/x86_64/ 上海交大

http://mirrors.163.com/centos/7/isos/x86_64/ 网易
http://mirrors.cn99.com/centos/7/isos/x86_64/ 网易开源镜像站
http://mirrors.sohu.com/centos/7/isos/x86_64/ 搜狐
~~~

- 升级内核：[如何在 CentOS 7 中安装或升级最新的内核](https://linux.cn/article-8310-1.html)

## 常用命令

- 安装软件

~~~shell
yum list installed
yum install tomcat
yum remove tomcat
yum update tomcat
~~~

- 升级软件

~~~shell
# gcc 升级：https://blog.csdn.net/yzpbright/article/details/81571645
# 依赖安装：https://blog.51cto.com/liuzhengwei521/2112118
wget https://ftp.gnu.org/gnu/gcc/gcc-9.1.0/gcc-9.1.0.tar.gz
tar -xzvf gcc-9.1.0.tar.gz
~~~

- rpm安装

~~~shell
rpm -ivh your-package.rpm
~~~

## 错误处理

- (28, 'Operation too slow. Less than 1 bytes/sec transfered the last 30 seconds')

~~~wiki
/etc/yum.conf文件中添加，默认30，改成120
timeout=120
~~~
