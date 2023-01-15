# Usage of Python

- [Usage of Python](#usage-of-python)
  - [data load \& store](#data-load--store)
    - [`.txt`](#txt)
    - [`.tsv`](#tsv)
    - [`.pt`](#pt)
    - [`.bin`](#bin)
  - [Basic Data Structure](#basic-data-structure)
    - [dict](#dict)
    - [set](#set)
    - [list](#list)
  - [numpy](#numpy)
  - [Matplotlib](#matplotlib)
  - [Faker](#faker)

## data load & store

### `.txt`

```python
with open("xxx.txt", "w", encoding="utf8") as f:
    f.write("xxx\n123\n")

with open("xxx.txt", "r", encoding="utf8") as f:
    f.read() # xxx\n123\n

with open("xxx.txt", "r", encoding="utf8") as f:
    f.read(1) # x

with open("xxx.txt", "r", encoding="utf8") as f:
    f.readline() # xxx

with open("xxx.txt", "r", encoding="utf8") as f:
    f.readlines() # ['xxx\n', '123\n']
```

### `.tsv`

```python
import csv

with open("xxx.tsv", "w", encoding="utf8") as f:
    f.write("xxx\txxx\txxx\n")

with open("xxx.tsv", "r", encoding="utf8") as f:
    reader = csv.reader(f, delimiter="\t")
    for row in reader:
        ...
```

### `.pt`

```python
import pickle

with open("xxx.pt", 'wb') as f:
    pickle.dump((data1, data2), f)

with open("xxx.pt", 'rb') as f:
    data1, data2 = pickle.load(f)
```

**Warning: `pickle.load()`不信任的文件可能会带来安全隐患。** `pickle`是专门为python设计的序列化工具，它可以将python中的对象转换为字节流，并能复原为python对象。但是，python为`class`添加了一个特别的`__reduce__()` method用来告诉`pickle`如何复原数据，我们可以利用这一method执行不安全的代码。一个例子如下:

```python
import pickle
import subprocess

class Dangerous:
    def __reduce__(self):
        return (
            subprocess.Popen, 
            (('/bin/bash', "-c", "ls"),),
        )
d = Dangerous()

with open("data.pt", 'wb') as f:
    pickle.dump(d, f)

with open("data.pt", 'rb') as f:
    data = pickle.load(f)
```

执行上述代码，在`load pickle`文件时，会执行`Dangerous`的`__reduce__`method用于恢复数据，在上例中就是打开了`bash`并执行`ls`命令。可以发现，如果随意加载`pickle`文件，可能会带来安全隐患。

### `.bin`

```python
import struct

with open("xxx.bin", 'wb') as f:
    data = [1.0, 2.0, 3.0]
    length = len(data)
    # struct.pack(format, data)
    f.write(struct.pack("i", length))
    f.write(struct.pack("%sf" * length, *data))

with open("xxx.bin", 'rb') as f:
    # struct.unpack(format, buffer)
    i = struct.unpack("i", f.read(4))[0]
    _list = struct.unpack("%sf" % i, f.read(4 * i))[0]
```

## Basic Data Structure

### dict

```python
d = {"a": [1, 2], "b": [3], "c": [4, 5, 6]}

# insert
if key not in d.keys():
    d[key] = [value]
else:
    d[key].append(value)

# delete
for key in list(d.keys()):
    if d[key] == value:
        del d[key]

# traverse
for key in d.keys():
    print(key, d[key])

# str to dict
import json
str1 = "{'a': 1, 'b': 2}"
json.loads(str1)
```

### set

```python
a = set([1, 2, 3])
b = set([2, 3, 4])

# 求交集(计算recall)
a.intersection(b) # {2, 3}

# 求并集
a.union(b) # {1, 2, 3, 4}

# 求差集
a.difference(b) # {1}
b.difference(a) # {4}
```

### list

```python
a = ['1', '2', '3']
b = list(map(int, a)) # [1, 2, 3]
```

## numpy

```python
#设置随机种子
np.random.seed(0)
# 依概率p从data中随机采样size个数据
r = np.random.choice(data, p=p, size=10)
# 随机生成0~1之间的浮点数矩阵
r = np.random.random((10, 10))
# 随机生成[0, 9]整数矩阵
r = np.random.randint(0, 10, (10, 10))


# find index of a given value in the array
a = np.array([1, 3, 3, 4, 5])
index = np.where(a == 3)[0]     # [1, 2]
b = a[index]                    # [3, 3]

# L2 norm
x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
#按行计算
np.linalg.norm(x, axis=1)       # [3.74165739, 8.77496439, 13.92838828]
#按列计算
np.linalg.norm(x, axis=0)       # [9.53939201, 11.22497216, 12.12435565]

# arg sort
a = np.array([4, 2, 1, 5, 3])
np.argsort(a)           
# [2 1 4 0 3]
# first element is "1" which is in a[2]

# delete column/row
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
np.delete(a, 1, axis=0) # [1, 2, 3], [7, 8, 9]
np.delete(a, 1, axis=1) # [1, 3], [4, 6], [7, 9]
```

## Matplotlib

```python
import matplotlib.pyplot as plt
import numpy as np

#设置全局字体
plt.rc('font', family='Times New Roman')

#设置标题
plt.title('Test', fontsize=20, color='black')
# 设置坐标轴标签
plt.xlabel('axis_x', fontsize=15, color='black')
plt.ylabel('axis_y', fontsize=15, color='black')
# 设置刻度范围
plt.xlim(-10.0, 10.0)
plt.ylim(0.0, 10000.0)
#设置刻度scale
plt.yscale('log')
# 设置刻度标签
plt.xticks(np.arange(11), ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十'], fontsize=10, color='gray')
plt.yticks(fontsize=10, color='gray')

#画曲线
plt.plot(x, y, color='red', linewidth=1.0, linestyle='--', marker='.', label='label1')
"""
color: 颜色
linewidth: 线宽
linestyle: 线型
marker: 标记样式
label: 图例
"""

#画散点
plt.scatter(x, y, color='red', marker='o', label='label2', s=10, alpha=0.6)
"""
color: 颜色
marker: 标记样式
label: 图例
s: 标记大小
alpha: 透明度
"""

#画柱状图
num_of_algo = 3     #参与绘图的算法数目
num_of_data = 5     #数据数目
bar_width = 0.30    #柱宽

#设置每个柱子的x坐标
index = np.arange(0, num_of_data, 1)

# 画柱状图 (data是 num_of_algo*num_of_data 的矩阵)
for i in range(num_of_algo):
    plt.bar(index + i * bar_width, data[i], bar_width, label=label[i], facecolor=facecolor[i], edgecolor=edgecolor[i], hatch=hatch[i])
    """
    index + i * bar_width: 柱子的x坐标
    data[i]: 柱子的高度
    bar_width: 柱宽
    label: 图例
    facecolor: 柱子填充颜色
    edgecolor: 柱子边框颜色
    hatch: 柱子填充样式
    """
    for a, b in zip(index + i * bar_width, data[i]):
        # a: 文字的x坐标，b: 文字的y坐标
        plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=5, rotation=90)

# 设置x轴刻度在中间
plt.xticks(index + (num_of_algo-1)*bar_width / 2,  index)



plt.legend(bbox_to_anchor=(0.9, 1.2), fontsize=30, ncol=2, markerscale=2, frameon=True)
"""
bbox_to_anchor: 图例位置
fontsize: 字体大小
ncol: 列数
markerscale: 标记大小
frameon: 是否显示边框
"""

#紧致布局
plt.tight_layout()
#保存为矢量图
plt.savefig("multicol_traverse.svg", format="svg")
plt.show()
```

## Faker

install:

```bash
pip install faker
```

usage: https://www.jianshu.com/p/6bd6869631d9


```python
from faker import Faker

fake = Faker()
location_list = [fake.country() for _ in range(200)]
```