# pip安装问题

- windows下安装时编码问题

~~~
打开 
c:\program files\python36\lib\site-packages\pip\compat\__init__.py 约75行 
return s.decode('utf_8') 改为return s.decode('cp936')

原因： 
编码问题，虽然py3统一用utf-8了。但win下的终端显示用的还是gbk编码。
~~~