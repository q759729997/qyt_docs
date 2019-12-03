# GIT免密设置

- 参考：<https://blog.csdn.net/chenle408/article/details/57120999>
- 设置Git的user name和email：

~~~
$ git config --global user.name "759729997"
$ git config --global user.mail "759729997@qq.com"
~~~

- 生成SSH密钥过程：

~~~
1.查看是否已经有了ssh密钥：cd ~/.ssh
如果没有密钥则不会有此文件夹，有则备份删除
2.生存密钥：

$ ssh-keygen -t rsa -C "759729997@qq.com"
按3个回车，密码为空。
~~~

- 这样即可在C:\Users\admin\.sss文件下得到两个文件：id_rsa 和id_rsa.pub;

- 登录https://github.com/，如没有账户则注册登录进入。

~~~
登录进去之后，选择SSH and GPG Keys->New SSH key
Key的内容为id_rsa.pub里面的内容（可用word打开）
Title的内容可以自己任意指定
~~~

- 测试：ssh git@github.com

~~~

$ sshgit@github.com

Theauthenticity of host 'github.com (192.30.253.112)' can't be established.

RSA keyfingerprint is SHA256:nThbg6kXUpJWGl7E1IGOCspRomTxdCARLviKw6E5SY8.

Are yousure you want to continue connecting (yes/no)? yes

Warning:Permanently added 'github.com,192.30.253.112' (RSA) to the list of known hosts.

PTY allocationrequest failed on channel 0

Hichenle90! You've successfully authenticated, but GitHub does not provide shellaccess.

Connectionto github.com closed.

~~~