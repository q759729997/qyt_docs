# jupyter简易教程

## 简介

- Jupyter Notebook（此前被称为 IPython notebook）是一个交互式笔记本，支持运行 40 多种编程语言。
- Jupyter Notebook 的本质是一个 Web 应用程序，便于创建和共享文学化程序文档，支持实时代码，数学方程，可视化和 markdown。 用途包括：数据清理和转换，数值模拟，统计建模，机器学习等等 。（来源于百度百科）

## 启动

~~~python
jupyter notebook --allow-root //启动,allow-root为root用户访问时需要添加的命令
~~~

## 安装配置

- 安装与卸载

~~~python
pip install notebook -i https://pypi.douban.com/simple/ //安装
pip uninstall notebook  //卸载
jupyter notebook --generate-config //生成配置
pip install pyrsistent==0.15.0  //windows安装
~~~

- 远程访问配置，以下设置可以保证远程访问服务器上的jupyter,**xshell工具可以通过隧道设置端口转发**，本地就可以使用localhost进行访问。**本地访问时，使用localhost加对应端口进行访问，输入启动页面对应的token即可，密码设置也可在此web页面进行。**

~~~python
c.NotebookApp.ip = '*'  # ip配置，防止Cannot assign requested address
c.NotebookApp.port = 8765  # port配置
c.NotebookApp.notebook_dir = '/root/notebook'  # 默认启动路径
c.NotebookApp.open_browser = False  # 是否打开浏览器
~~~

