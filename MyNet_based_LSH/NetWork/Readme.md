# Generate Data
生成分类数据和回归数据（都是随机的），以及数据的reshape，归一化操作  
和 loss/accuracy 文件画图(csv 保存)  
> 之后的训练采用的AI领域的一些经典数据集

# Auto diff
> 自动微分实现  
> 基于计算图实现了自动微分，从单个结点类似地拓展到矩阵运算  
> 之后也会继续补充  hh  
> 实现采用的 eager 模式，所以 execute 相当于跑了两次  
> ！tensorflow 采用的是lazy模式

主要函数  
- 利用 identity 函数初始化矩阵节点
- 很多种的 Op 操作，有空会继续更新
- execute构建完表达式之后通过 gradients 方法进行反向传播更新

# Net
主要思路还是模拟 pytorch，写一个非常简单版的 minitorch
先构建网络，然后往里面添加不同的层  
前向后向通过之前实现的 execute 函数跑完  
每个矩阵节点都能够得到相应的梯度函数  
前向的话是通过 inputs 去跑  
反向传播是通过记录的 loss 列表去跑  

# Layer
从 Net 降到 layer 来看，layer 就是一系列 matrix 节点操作构成的序列组合  
多个层构成 Net  
Net 进行 forward 前向传播时，会从 input 延层往后传  
backward 时，从记录的 loss 列表逐个反向传播经过层  

# LogAndLoss
简单的loss saver类和 Log 类，记录和打印日志用的。。。

# main
整个文件跑的测试

# lab.ipynb
main 拆开 debug 用的



