## 任务名称

线性回归；Softmax与分类模型、多层感知机

## 学习心得

1. 对pytorch神经网络的基本搭建进行了复习巩固。

2. 批量的损失函数平均值计算公式中，除以2是为了简化求导运算：

![img](https://upload-images.jianshu.io/upload_images/8518346-69ff119665ead6ac.png?imageMogr2/auto-orient/strip|imageView2/2/w/687/format/webp)

回归损失函数

3. 在pytorch中view函数的作用为重构张量的维度，相当于numpy中resize()的功能。

   参数中的-1就代表这个位置由其他位置的数字来推断，只要在不致歧义的情况的下，view参数就可以推断出来.

   view(-1)将张量变为一维结构，即展平。