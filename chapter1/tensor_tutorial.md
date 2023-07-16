```python
%matplotlib inline
```


什么是Pytorch
================

这是一个基于Python的科学计算软件包，面向两组受众：

-  替代NumPy以使用GPU的功能
-  深度学习研究平台，可提供最大的灵活性和速度

开始吧
---------------

## 张量(Tensors)

张量与NumPy的ndarrays类似，此外，  
张量也可以在GPU上使用以加速计算。




```python
from __future__ import print_function
import torch
```

创建一个未初始化的5x3矩阵：




```python
x = torch.empty(5, 3)
print(x)
```

    tensor([[-1.8736e-02,  5.3810e-43, -1.8736e-02],
            [ 5.3810e-43, -1.8736e-02,  5.3810e-43],
            [-1.8736e-02,  5.3810e-43, -1.8736e-02],
            [ 5.3810e-43, -1.8736e-02,  5.3810e-43],
            [-1.8736e-02,  5.3810e-43, -1.8736e-02]])
    

创建一个随机初始化的矩阵：




```python
x = torch.rand(5, 3)
print(x)
```

    tensor([[0.1884, 0.9676, 0.0932],
            [0.0237, 0.8706, 0.1165],
            [0.8923, 0.6846, 0.2428],
            [0.1050, 0.1057, 0.4467],
            [0.2271, 0.8529, 0.5744]])
    

构造一个填充零且dtype long的矩阵：




```python
x = torch.zeros(5, 3, dtype=torch.long)
print(x)
```

    tensor([[0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]])
    

直接从数据创建张量：




```python
x = torch.tensor([5.5, 3])
print(x)
```

    tensor([5.5000, 3.0000])
    

或基于现有张量创建张量。 这些方法将重复使用输入张量的属性，例如dtype，除非用户提供新值




```python
x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
print(x)

x = torch.randn_like(x, dtype=torch.float)    # override dtype!
print(x)                                      # result has the same size
```

    tensor([[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]], dtype=torch.float64)
    tensor([[-0.6246, -0.3308,  1.0961],
            [-0.2643, -1.9371, -0.7324],
            [-1.0433,  1.8659, -0.2630],
            [-0.5869, -0.8914,  1.2099],
            [-0.7889,  0.0762,  0.3997]])
    

打印它的大小：




```python
print(x.size())
```

    torch.Size([5, 3])
    

<div class="alert alert-info"><h4>注</h4><p>

``torch.Size`` 实际上是一个tuple，因此它支持所有tuple操作。</p></div>

## 运算(Operations)

运算有多种语法实现。在下面的示例中，我们将看一下加法运算。

加法：语法1




```python
y = torch.rand(5, 3)
print(x + y)
```

    tensor([[-0.5411, -0.2331,  1.4236],
            [ 0.4124, -1.0028, -0.6942],
            [-0.5186,  2.3691,  0.6042],
            [-0.3107, -0.7324,  1.7120],
            [-0.5278,  0.2224,  1.1851]])
    

加法：语法2




```python
print(torch.add(x, y))
```

    tensor([[-0.5411, -0.2331,  1.4236],
            [ 0.4124, -1.0028, -0.6942],
            [-0.5186,  2.3691,  0.6042],
            [-0.3107, -0.7324,  1.7120],
            [-0.5278,  0.2224,  1.1851]])
    

加法：提供输出张量作为参数(argument)




```python
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)
```

    tensor([[-0.5411, -0.2331,  1.4236],
            [ 0.4124, -1.0028, -0.6942],
            [-0.5186,  2.3691,  0.6042],
            [-0.3107, -0.7324,  1.7120],
            [-0.5278,  0.2224,  1.1851]])
    

加法：就地(in-place)




```python
# adds x to y
y.add_(x)
print(y)
```

    tensor([[-0.5411, -0.2331,  1.4236],
            [ 0.4124, -1.0028, -0.6942],
            [-0.5186,  2.3691,  0.6042],
            [-0.3107, -0.7324,  1.7120],
            [-0.5278,  0.2224,  1.1851]])
    

<div class="alert alert-info"><h4>注</h4><p>

任何使张量就地发生变化的操作都将使用 ``_``.
    如: ``x.copy_(y)``, ``x.t_()``, 会改变 ``x``.</p></div>

您可以使用类似NumPy的标准索引来变出各种花样！




```python
print(x[:, 1])
```

    tensor([-0.3308, -1.9371,  1.8659, -0.8914,  0.0762])
    

调整大小：如果要调整张量的大小/形状，可以使用 ``torch.view``:




```python
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())
```

    torch.Size([4, 4]) torch.Size([16]) torch.Size([2, 8])
    

如果您具有一个元素张量，请使用``.item()``获取该值作为Python数字。




```python
x = torch.randn(1)
print(x)
print(x.item())
```

    tensor([0.5033])
    0.5032538771629333
    

**稍后阅读:**

  [这里](https://pytorch.org/docs/torch)
  
包含了100多个Tensor运算，包括转置(transposing)、索引(indexing)、分割(slicing)、数学运算(mathematical operations)、线性代数(linear algebra)、随机数(random numbers)等。

NumPy转换
------------

将Torch张量转换为NumPy数组，反之亦然，这十分简单。

Torch张量和NumPy数组将共享其基础内存位置，并且更改一个将更改另一个。

## 将Torch张量转换为NumPy数组




```python
a = torch.ones(5)
print(a)
```

    tensor([1., 1., 1., 1., 1.])
    


```python
b = a.numpy()
print(b)
```

    [1. 1. 1. 1. 1.]
    

看看numpy数组的值如何变化。




```python
a.add_(1)
print(a)
print(b)
```

    tensor([2., 2., 2., 2., 2.])
    [2. 2. 2. 2. 2.]
    

## 将NumPy数组转换为Torch张量

查看更改numpy数组如何自动更改Torch Tensor




```python
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)
```

    [2. 2. 2. 2. 2.]
    tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
    

除CharTensor之外，CPU上的所有张量都支持转换为NumPy并转回。

CUDA张量
------------

张量可以使用``.to``方法移动到任何设备上。



```python
# 让我们仅在CUDA可用时运行此单元格
# 我们将使用``torch.device``对象将张量移入和移出GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # CUDA设备对象
    y = torch.ones_like(x, device=device)  # 在GPU上直接创建张量
    x = x.to(device)                       # 或只使用字符串``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` 也可以一起改变dtype！
```

    tensor([1.5033], device='cuda:0')
    tensor([1.5033], dtype=torch.float64)
    
