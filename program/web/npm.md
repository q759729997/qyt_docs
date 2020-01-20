## 简介

- 安装包查看页面：<https://npm.taobao.org/>
- npm使用：<https://www.runoob.com/nodejs/nodejs-npm.html>

## 命令

~~~python
npm init  # 查看项目信息
npm info express # 查看包的详细信息
npm root -g # 查看全局安装包的存放位置
~~~

- 项目编译安装

~~~python
npm install
npm start
~~~

- build项目

~~~python
npm install dotenv --global-style  # 在其他联网电脑上执行
将dotenv拷贝到当前项目node_modules内
其他依赖包：
npm install fs-extra --global-style
npm install filesize --global-style
npm install gzip-size --global-style
npm install webpack --global-style

npm run-script build
~~~

## 安装包

- 安装命令

~~~
npm config set registry https://registry.npm.taobao.org  # 更改 npm 的下载镜像为淘宝镜像
npm install 或者 npm install –save-dev会自动将package.json中的模块安装到node-modules文件夹下
npm install express -g  # 全局安装
npm install express # 本地安装在当前目录node_modules文件夹内
npm uninstall express  # 卸载
npm list -g  # 查看安装信息
npm list grunt  # 查看某个模块的版本号
~~~

- 离线安装

~~~
npm install rasa-nlu-trainer --global-style  # -global-style，表示将rasa-nlu-trainer安装到node_modules中一个单独的rasa-nlu-trainer文件夹中
npm install rasa-nlu-trainer/ -g  # 直接从文件夹安装
tar zvcf rasa-nlu-trainer.tar.gz rasa-nlu-trainer/  # 压缩
tar -xzvf rasa-nlu-trainer.tar.gz  # 解压
~~~