---

title: 模板匹配算法

lang: zh-CN

---

# 第一个模式识别算法-模板匹配算法

## 什么是模板匹配算法 Template Matching? 

> 模板匹配是一种最原始、最基本的模式识别方法，研究某一特定对象物的图案位于图像的什么地方，进而识别对象物，这就是一个匹配问题。它是图像处理中最基本、最常用的匹配方法。模板匹配具有自身的局限性，主要表现在它只能进行平行移动，若原图像中的匹配目标发生旋转或大小变化，该算法无效。

模板匹配是一种高级的计算机视觉技术，可识别图像上与预定义模板匹配的部分。它是在整个图像上移动模板并计算模板与图像上被覆盖窗口之间的相似度的过程。模板匹配是通过二维卷积实现的。在卷积中，输出像素的值通过将两个矩阵的元素相乘并对结果求和。其中一个矩阵代表图像本身，另一个矩阵是模板，为卷积核。

模板就是一副已知的小图像，而模板匹配就是在一副大图像中搜寻目标，已知该图中有要找的目标，且该目标同模板有相同的尺寸、方向和图像元素，通过一定的算法可以在图中找到目标，确定其坐标位置。

模板匹配算法在很多库中可以直接使用例如，**OpenCV**，**scikit-learn**，下面我们将会通过代码实现的角度来简单解释模板匹配的原理。

## 代码实现

### 1. 准备一张包含待匹配模板的原始图像和一幅待匹配的目标图像。

``````python
import cv2
from PIL import Image 
# 加载原始图像和模板图像
original_image = cv2.imread('input/origin_image.png',3)
template_image = cv2.imread('input/template_image.png')

original_image_gray = cv2.cvtColor(original_image,cv2.COLOR_RGB2GRAY)
template_image_gray = cv2.cvtColor(template_image,cv2.COLOR_RGB2GRAY)

import matplotlib.pyplot as plt

from matplotlib import rcParams
rcParams['font.family'] = 'SimHei'

fig, ax = plt.subplots(2,2,figsize=(9,9))
ax[0][0].imshow(original_image[:,:,::-1])
ax[0][0].set_title("原图像")
ax[0][1].imshow(original_image_gray,cmap='gray')
ax[1][0].imshow(template_image[:,:,::-1])
ax[1][0].set_title("模板图像")
ax[1][1].imshow(template_image_gray,cmap='gray')

``````

![image-20230721184542738](/template-matching/image-20230721184542738.png)

### 2. 选择一种适当的相似度度量方法（如均方差、相关性、归一化互相关等）作为匹配度量指标。

首先尝试使用标准相关性系数匹配作为匹配度量指标。

```python
method = cv2.TM_CCOEFF_NORMED
```

这里的匹配度量指标也有其他选择： **(归一化结果更可靠）**

* TM_SQDIFF：计算平方不同，得出的值越小，越相关

* TM_CCORR：计算相关性，得出的值越大，越相关

* TM_CCOEFF：计算相关系数，得出的值越大，越相关

* TM_SQDIFF_NORMED：计算归一化平方不同，得出的值越接近零，越相关

* TM_CCORR_NORMED：计算归一化相关性，得出的值越接近1，越相关

* TM_CCOEFF_NORMED：计算归一化相关系数，得出的值越接近1，越相关

### 3. 定义一个滑动窗口，以模板的大小为窗口尺寸，在目标图像上滑动窗口，并计算窗口与模板的匹配度量值。

```python
result = cv2.matchTemplate(original_image_gray, template_image_gray, method)
fig, ax = plt.subplots(1,2,figsize=(9,9))
ax[0].imshow(original_image[:,:,::-1])
ax[1].imshow(result)
```

![image-20230721185103480](/template-matching/image-20230721185103480.png)

### 4. 选择匹配度量值最小/最大的位置作为模板在目标图像中的匹配位置。

```python
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
print("maxVal",maxVal)
print("maxLoc",maxLoc)
```

| 输出   |                    |
| ------ | ------------------ |
| maxVal | 0.9580235481262207 |
| maxLoc | (100, 244)         |

### 5. 对匹配结果进行可视化，例如在图像上绘制矩形框标识匹配位置。

```python
output_image = cv2.imread('input/origin_image.png',3)

(startX, startY) = maxLoc
endX = startX + template_image_gray.shape[1]
endY = startY + template_image_gray.shape[0]

cv2.rectangle(output_image, (startX, startY), (endX, endY), (0, 255, 0), 2)

fig, ax = plt.subplots(1,2,figsize=(9,6))
fig.suptitle("最大匹配值结果")
ax[0].imshow(output_image[:,:,::-1])
ax[1].imshow(template_image[:,:,::-1])
```

![image-20230721185525890](/template-matching/image-20230721185525890.png)

### 6. 重复以上步骤，可以尝试不同的模板、不同的相似度度量方法或其他参数调整来观察实验结果的变化。

#### 6.1 尝试不同相似度度量方法

归一化互相关匹配

```python
result = cv2.matchTemplate(original_image_gray, template_image_gray, cv2.TM_CCORR_NORMED)
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
print("maxVal",maxVal)
print("maxLoc",maxLoc)
```

| 输出   |                   |
| ------ | ----------------- |
| maxVal | 0.995827317237854 |
| maxLoc | (100, 244)        |

![image-20230721185718853](/template-matching/image-20230721185718853.png)

#### 6.2 尝试不同相似度阈值

```python
import numpy as np
output_image = cv2.imread('input/origin_image.png',3)
w = template_image_gray.shape[1]
h = template_image_gray.shape[0]
threshold = 0.9
loc = np.where(result >= threshold)
for pt in zip(*loc[::-1]):
    bottom_right = (pt[0] + w, pt[1] + h)
    cv2.rectangle(output_image, pt, bottom_right, (0, 255, 0), 2)

fig, ax = plt.subplots(1,2,figsize=(9,6))
fig.suptitle("匹配度>0.9")
ax[0].imshow(output_image[:,:,::-1])
ax[1].imshow(template_image[:,:,::-1])
```

![image-20230721185806660](/template-matching/image-20230721185806660.png)

![image-20230721185820829](/template-matching/image-20230721185820829.png)

#### 6.4 尝试不同模板图像

```python
template2_image = cv2.imread('input/template2_image.png')
template2_image_gray = cv2.cvtColor(template2_image,cv2.COLOR_RGB2GRAY)
result = cv2.matchTemplate(original_image_gray, template2_image_gray, cv2.TM_CCOEFF_NORMED)
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
print("maxVal",maxVal)
print("maxLoc",maxLoc)
output_image = cv2.imread('input/origin_image.png',3)

(startX, startY) = maxLoc
endX = startX + template2_image_gray.shape[1]
endY = startY + template2_image_gray.shape[0]

cv2.rectangle(output_image, (startX, startY), (endX, endY), (0, 255, 0), 2)

fig, ax = plt.subplots(1,2,figsize=(9,6))
fig.suptitle("最大匹配值结果")
ax[0].imshow(output_image[:,:,::-1])
ax[1].imshow(template2_image[:,:,::-1])
```

![image-20230721185910894](/template-matching/image-20230721185910894.png)

```python
output_image = cv2.imread('input/origin_image.png',3)
threshold = 0.2
loc = np.where(result >= threshold)
w = template2_image_gray.shape[1]
h = template2_image_gray.shape[0]
for pt in zip(*loc[::-1]):
    bottom_right = (pt[0] + w, pt[1] + h)
    cv2.rectangle(output_image, pt, bottom_right, (0, 255, 0), 2)
fig, ax = plt.subplots(1,2,figsize=(9,6))
fig.suptitle("匹配度>0.2")
ax[0].imshow(output_image[:,:,::-1])
ax[1].imshow(template2_image[:,:,::-1])
```

![image-20230721185941177](/template-matching/image-20230721185941177.png)