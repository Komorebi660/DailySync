# Usage of Python

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

## numpy

