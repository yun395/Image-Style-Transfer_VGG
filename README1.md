# 第10章 图像风格迁移

##  10.1 VGG模型

采用简单粗暴的堆砌3*3卷积层的方式构建模型

结构简单，容易理解，便于利用到其他任务当中去

VGG-19网络的卷积部分由5个卷积块构成，每个卷积块中有多个卷积层，结尾处有一个池化层

![img](https://i-blog.csdnimg.cn/direct/ee1e0166fd8d44b1aa8c2485202174ee.png)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)编辑

## 10.2 图像风格迁移介绍

两种特征度量：一种用于表示图像的内容，另一种用于表示图像的风格

底层卷积层提取的风格特征较细节，提取的内容特征较详细；高层卷积层提取的风格特征较整体，提取的内容特征较概括

## 10.3 内容损失函数

内容损失函数主要利用卷积中两图的特征图相减

### 10.3.1 定义

用于衡量两幅图像之间的内容差异大小

![L_{c}=\frac{1}{2}\sum_{i,j}^{}(X_{i,j}^{l}-Y_{i,j}^{l})^{2}](https://latex.csdn.net/eq?L_%7Bc%7D%3D%5Cfrac%7B1%7D%7B2%7D%5Csum_%7Bi%2Cj%7D%5E%7B%7D%28X_%7Bi%2Cj%7D%5E%7Bl%7D-Y_%7Bi%2Cj%7D%5E%7Bl%7D%29%5E%7B2%7D)

其中，![X^{l}](https://latex.csdn.net/eq?X%5E%7Bl%7D)和![Y^{l}](https://latex.csdn.net/eq?Y%5E%7Bl%7D)分别是两幅图片由VGG网络某一卷积层提取的特征图，l表示卷积层的下标，i和j表示矩阵中行与列的下标

两幅图片的内容损失函数是由特征图对位求差得到的

第4个卷积块的第二层（conv4_2）用于计算内容损失

### 10.3.2 内容损失模块的实现

## 10.4 风格损失函数

### 10.4.1 风格损失函数的定义

计算特征图的Gram矩阵得到图像风格的数学表示

Gram矩阵：![G_{i,j}^{l}=\sum_{k}^{}X_{ik}^{l}X_{jk}^{l}](https://latex.csdn.net/eq?G_%7Bi%2Cj%7D%5E%7Bl%7D%3D%5Csum_%7Bk%7D%5E%7B%7DX_%7Bik%7D%5E%7Bl%7DX_%7Bjk%7D%5E%7Bl%7D)，表示的是特征与特征（卷积核与卷积核）的相关性

风格损失函数：

![L_{s}^{l}=\frac{1}{4N_{l}^{2}M_{l}^{2}}\sum_{i,j}^{}(G_{ij}^{l}H_{ij}^{l})^{2}](https://latex.csdn.net/eq?L_%7Bs%7D%5E%7Bl%7D%3D%5Cfrac%7B1%7D%7B4N_%7Bl%7D%5E%7B2%7DM_%7Bl%7D%5E%7B2%7D%7D%5Csum_%7Bi%2Cj%7D%5E%7B%7D%28G_%7Bij%7D%5E%7Bl%7DH_%7Bij%7D%5E%7Bl%7D%29%5E%7B2%7D)

![L_{s}=\sum_{l}^{}w_{l}L_{s}^{l}](https://latex.csdn.net/eq?L_%7Bs%7D%3D%5Csum_%7Bl%7D%5E%7B%7Dw_%7Bl%7DL_%7Bs%7D%5E%7Bl%7D)

![N_{l}](https://latex.csdn.net/eq?N_%7Bl%7D) 和![M_{l}](https://latex.csdn.net/eq?M_%7Bl%7D)分别为特征图的通道数和边长，![w_{l}](https://latex.csdn.net/eq?w_%7Bl%7D)为权重

conv1_1,conv2_1,conv3_1,conv4_1,conv5_1 用于计算风格损失

### 10.4.2 计算Gram矩阵函数的实现

### 10.4.3 风格损失模块的实现

## 10.5 优化过程 

总损失函数：

![L=\alpha L_{c}+\beta L_{s}](https://latex.csdn.net/eq?L%3D%5Calpha%20L_%7Bc%7D&plus;%5Cbeta%20L_%7Bs%7D)

最小化以上损失函数

使用L-BFGS算法进行优化

## 10.6 图像风格迁移主程序的实现

### 10.6.1 图像预处理

### 10.6.2 参数定义

### 10.6.3 模型初始化

### 10.6.4 运行风格迁移的主函数

### 10.6.5 利用VGG网络建立损失函数

### 10.6.6 风格迁移的优化过程

### 10.6.7 运行风格迁移

