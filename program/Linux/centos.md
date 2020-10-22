# centos

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
