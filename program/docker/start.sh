#/bin/bash
echo "=============进入主目录============="
# rm /bin/sh
# ln -s /bin/bash /bin/sh
cd /root/projects
pwd
echo "=============设置Python环境变量============="
source activate python37
python --version
# # 查看现有外置Python环境，若与下面的export的目录一致，则可直接跳过后面的unset与export步骤
# echo $PYTHONPATH
# # 删除原有外置Python环境
# unset PYTHONPATH
# # 设置新的Python环境
# export PYTHONPATH=$PYTHONPATH:/root/projects/python_packages/kdTimeConvert
# export PYTHONPATH=$PYTHONPATH:/root/projects/python_packages/kdChatbot
# # 查看新的外置Python环境，如果与export的目录一致，则代表成功
# echo $PYTHONPATH
# echo "=============进入项目目录============="
# cd /root/projects/chatbot_guodiao_automation
# pwd
# echo "=============启动服务============="
# sh scripts/start_server.sh
# sh scripts/start_interface_server.sh
echo "=============保持容器不关闭============="
# tail -f log_nlu_server.log
while true;do sleep 10;done
