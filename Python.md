# Usage of Python

- [Usage of Python](#usage-of-python)
  - [data load \& store](#data-load--store)
    - [`.tsv`](#tsv)
    - [`.pt`](#pt)
    - [`.bin`](#bin)
  - [Basic Data Structure](#basic-data-structure)
    - [dict](#dict)
    - [set](#set)
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

## numpy

```python
# find index of a given value in the array
a = np.array([1, 2, 3, 4, 5])
np.where(a == 3)[0]     # 2

# L2 norm
np.linalg.norm(a)       # 7.416198487095663

# arg sort
a = np.array([4, 2, 1, 5, 3])
np.argsort(a)           
# [2 1 4 0 3]
# first element is "1" which is in a[2]
```