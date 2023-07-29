---

title: SVM算法的应用

lang: zh-CN

---

# SVM算法的应用

## 什么是SVM(Support Vector Machine)算法? 

> SVM(Support Vector Machine)算法，即支持向量机算法，它是最优秀的分类算法之一，也是数据挖掘十大算法之一，它以其简单的理论构造了复杂的算法，又以其简单的用法实现了复杂的问题而受到业界的青睐。SVM算法属于有监督学习算法。它是在1995年由Corinna Cortes和Vapnik首先提出的。

SVM的核心思想可以归纳为以下两点：

1) 它是针对线性可分情况进行分析，对于线性不可分的情况，通过使用非线性映射算法将低维输入空间线性不可分的样本转化为高维特征空间使其线性可分，从而使得高维特征空间采用线性算法对样本的非线性特征进行线性分析成为可能。
2) 它以结构风险最小化理论为基于在特征空间中构建最优分割超平面，使得学习器得到全局最优化，并且在整个样本空间的期望风险以某个概率满足一定上界。

最后，简单解释一下支持向量机的含义，支持向量的含义是指支持 or 支撑平面上把两类类别划分开来的超平面的向量点，“机”的意思就是算法。

![01](/SVM/01.jpg)

如上图，w*x + b = 0 即为分类超平面，而红色的点为支持向量点，分类超平面可以将两类点进行分类。



## 代码实现

### 1. 数据准备

准备用于训练和测试的数据集，包括特征向量和对应的类别标签

```python
import random
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

n_samples = 100
X1 = np.random.randn(n_samples // 2, 2) - 1.5
X2 = np.random.randn(n_samples // 2, 2) + 1.5
dataset = np.vstack((X1, X2))

target = np.hstack((-np.ones(n_samples // 2), np.ones(n_samples // 2)))

plt.scatter(dataset[:, 0], dataset[:, 1], c=target)
plt.show()
```

![image-20230729142904600](/SVM/image-20230729142904600.png)

### 2. SVM算法实现

Python主流的机器学习库中都有实现SVM的方法，直接调用即可，在此处我们自己手动实现一下SVM的算法。

编写SVM算法的核函数和训练函数，以实现支持向量机分类器。

```python
class Supper_Vector_Machine:
    def __init__(self, dataset, target, C, toler, Iter_max):
        self.dataset = dataset
        self.target = target
        self.N, self.M = len(self.dataset), len(dataset[0])
        self.C = C
        self.toler = toler
        self.b = 0
        self.Alpha = np.zeros(self.N)
        self.iter_max = Iter_max
        self.w = np.zeros(self.M)
 
    def Fx(self, i):
        fxi = 0
        for k in range(self.N):
            fxi += self.Alpha[k] * self.target[k] * np.matmul(self.dataset[i], self.dataset[k].T)
        fxi += self.b
        return fxi
 
    def Kernel(self, i, j):
        result = np.matmul(self.dataset[i], self.dataset[j].T)
        return result
 
    def random_j(self, i):
        while True:
            j = random.choice(range(self.N))
            if j != i:
                return j
 
    def get_L_H(self, i, j):
        L, H = 0, 0
        if self.target[i] != self.target[j]:
            L = max([0, self.Alpha[j] - self.Alpha[i]])
            H = min([self.C, self.C + self.Alpha[j] - self.Alpha[i]])
        else:
            L = max([0, self.Alpha[j] + self.Alpha[i] - self.C])
            H = min([self.C, self.Alpha[i] + self.Alpha[j]])
        return L, H
 
    def filter(self, L, H, alpha_j):
        if alpha_j < L:
            alpha_j = L
        if alpha_j > H:
            alpha_j = H
        return alpha_j
 
    def SMO(self):
        iter = 0
        while iter < self.iter_max:
            change_num = 0
            for i in range(self.N):
                Fx_i = self.Fx(i)
                Ex_i = Fx_i - self.target[i]
                if self.target[i] * Ex_i < -self.toler and self.Alpha[i] < self.C or self.target[
                    i] * Ex_i > self.toler and self.Alpha[i] > 0:
                    j = self.random_j(i)
                    Fx_j = self.Fx(j)
                    Ex_j = Fx_j - self.target[j]
 
                    alpha_i = self.Alpha[i]
                    alpha_j = self.Alpha[j]
 
                    L, H = self.get_L_H(i, j)
                    if L == H:
                        continue
                    
                    eta = self.Kernel(i, i) + self.Kernel(j, j) - 2 * self.Kernel(i, j)
                    if eta <= 0:
                        continue
                    self.Alpha[j] += self.target[j] * (Ex_i - Ex_j) / eta
                    self.Alpha[j] = self.filter(L, H, self.Alpha[j])
 
                    if abs(self.Alpha[j] - alpha_j) < 0.00001:
                        continue
                    self.Alpha[i] += self.target[i] * self.target[j] * (alpha_j - self.Alpha[j])
                    b1 = self.b - Ex_i - self.target[i] * self.Kernel(i, i) * (self.Alpha[i] - alpha_i) - self.target[
                        j] * self.Kernel(i, j) * (self.Alpha[j] - alpha_j)
                    b2 = self.b - Ex_j - self.target[i] * self.Kernel(i, j) * (self.Alpha[i] - alpha_i) - self.target[
                        j] * self.Kernel(j, j) * (self.Alpha[j] - alpha_j)
 
                    if 0 < self.Alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.Alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2.0
                    change_num += 1
            if change_num == 0:
                iter += 1
            else:
                iter = 0
        for i in range(self.N):
            self.w += self.target[i] * self.Alpha[i] * self.dataset[i]
 
    def display(self):
        svm_point = []
        for i in range(100):
            if self.Alpha[i] > 0:
                svm_point.append(i)
        x_point = np.array([i[0] for i in dataset])
        y_point = np.array([i[1] for i in dataset])
        x = np.linspace(-4, 4, 5)
        y = -(self.w[0] * x + self.b) / self.w[1]
        p1 = plt.scatter(x_point[:50], y_point[:50], color='red')
        p2 = plt.scatter(x_point[-50:], y_point[-50:], color='blue')
        support_vector = np.array([dataset[i] for i in svm_point])
        p3 = plt.scatter(support_vector[:, 0], support_vector[:, 1], color='black')
        
        plt.legend((p1,p2,p3),('Class 1','Class 2', 'Support_vector') ,loc = 'best')     
        plt.plot(x, y)
        plt.show()

```

### 3. 模型训练

```python
model = Supper_Vector_Machine(dataset, target, 100, 0.01, 40)
model.SMO()
model.display()
```

![image-20230729143234048](/SVM/image-20230729143234048.png)

以上即实现了一个简单的SVM算法，蓝色县代表分类的超平面，黑色点则是找出的支持向量。