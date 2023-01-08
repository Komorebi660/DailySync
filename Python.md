# Usage of Python

- [Usage of Python](#usage-of-python)
  - [data load \& store](#data-load--store)
    - [`.tsv`](#tsv)
    - [`.pt`](#pt)
    - [`.bin`](#bin)
  - [Basic Data Structure](#basic-data-structure)
    - [dict](#dict)
    - [set](#set)
    - [list](#list)
  - [numpy](#numpy)

## data load & store

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
```

### set

```python
a = set([1, 2, 3])
b = set([2, 3, 4])

# 求交集(计算recall)
a.intersection(b) # {2, 3}
```

### list

```python
a = ['1', '2', '3']
b = list(map(int, a)) # [1, 2, 3]
```

## numpy

```python
# find index of a given value in the array
a = np.array([1, 3, 3, 4, 5])
np.where(a == 3)[0]     # [1, 2]

# L2 norm
np.linalg.norm(a)       # 7.416198487095663

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