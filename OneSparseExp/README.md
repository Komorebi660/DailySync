# Experiment for Paper *OneSparse*

## Step1: Get Data

下载并解压[MS Marco Passages](https://microsoft.github.io/msmarco/TREC-Deep-Learning-2019.html)数据集:

```bash
mkdir data
cd data
wget https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz
tar -zvxf collectionandqueries.tar.gz
```

我们只需要使用其中的三个文件:
- `collection.tsv`: 8,841,823 passages, 每一行是一篇passage, 格式为`pid \t passage`, `pid`从0开始，有序排列。
- `queries.dev.small.tsv`: 6,980 queries, 每一行是一个query, 格式为`qid \t query`, 无序排列。
- `qrels.dev.small.tsv`: query results.

读取`tsv`文件的代码如下:

```python
import csv

with open("xxx.tsv", "r", encoding="utf8") as f:
    reader = csv.reader(f, delimiter="\t")
    for row in reader:
        ...
```

接下来需要获取embedding数据, 可以用`scp`传输到服务器上。embedding数据包含两部分:

- `corpus`: 均匀拆分成了10份, 从`split00.pt`到`split09.pt`, 格式为`[embedding_matrix, pid_list]`, embedding用ar2g生成, 768维, inner product.
- `query`: 格式同上, `embedding_matrix`是6980*768的矩阵。

读取`pt`文件的代码如下:

```python
import pickle

with open("xxx.pt", 'rb') as f:
    embeddings, ids = pickle.load(f)
    # embeddings[0].tolist() 第一个embedding
    # int(ids[0]) 第一个id
```

### inner product -> l2 norm

inner product不满足三角不等式, 在query时可能会出现奇怪情况, [这篇文章](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances#how-can-i-do-max-inner-product-search-on-indexes-that-support-only-l2)提到了一种通过将embedding添加一维的方法, 将inner product转换为l2 norm.

```python
# for corpus embeddings
def transform(x): 
    '''x: np.array, shape=(8841823, 768)'''
    norms = np.linalg.norm(x, axis=1)**2
    phi = norms.max()
    extracol = np.sqrt(phi - norms)
    return np.hstack((extracol.reshape(-1, 1), x)).astype(np.float32)

# for query embeddings
def transform(x): 
    '''x: np.array, shape=(6980, 768)'''
    extracol = np.zeros(x.shape[0]).astype(np.float32)
    return np.hstack((extracol.reshape(-1, 1), x)).astype(np.float32)
```

## Step2: Elasticsearch

官方文档:

- [create index](https://www.elastic.co/guide/en/elasticsearch/reference/8.4/indices-create-index.html)
- [insert data](https://www.elastic.co/guide/en/elasticsearch/reference/8.4/docs-index_.html)
- [update data](https://www.elastic.co/guide/en/elasticsearch/reference/8.4/docs-update.html)
- [search](https://www.elastic.co/guide/en/elasticsearch/reference/8.4/search-search.html#request-body-search-query)

### docker安装

```bash
docker pull docker.elastic.co/elasticsearch/elasticsearch:8.4.3

#设置用户
docker network create elastic

#9200是es对外的默认端口, 这里把它转发到9400
docker run --name cyq-es --net elastic -p 9400:9200 -p 9500:9300 -e "discovery.type=single-node" -it docker.elastic.co/elasticsearch/elasticsearch:8.4.3
```

等待docker创建完毕，屏幕上会出现password, 将其记下来, 同时还需要复制`http_ca.crt`.

```bash
docker cp cyq-es:/usr/share/elasticsearch/config/certs/http_ca.crt .
```

验证certificate:

```bash
curl --cacert http_ca.crt -u elastic https://localhost:9400
```

输入密码后可以看到user的信息, 说明认证正常。之后可以关闭这一终端, 此时docker仍在后台运行, 可以使用`docker ps`查看。

我们可以通过`python request`访问es:

```python
import requests

user = 'elastic'
password = '**************'
cert = 'path-to-http_ca.crt'

url = "https://localhost:9400/xxx"
headers = {
        'Content-Type': 'application/json'
    }
payload = {
    ......
}

response = requests.request("PUT or POST", url, headers=headers, data=json.dumps(payload), verify=cert, auth=(user, password))
assert response.ok
```

### build index

代码见[此](./Elasticsearch/build-index.py)

```bash
#create index
nohup python3 -u build_index.py \
--create-index true \
--index-name "ms-marco" \
--path-doc-embedding "../embedding_data/corpus/split00.pt" \
--path-doc "../data/collection.tsv" > 0.log 2>&1 &

#insert data & build index
nohup python3 -u build_index.py \
--create-index false \
--index-name "ms-marco" \
--path-doc-embedding "../embedding_data/corpus/split01.pt" \
--path-doc "../data/collection.tsv" > 1.log 2>&1 &

nohup python3 -u build_index.py \
--create-index false \
--index-name "ms-marco" \
--path-doc-embedding "../embedding_data/corpus/split02.pt" \
--path-doc "../data/collection.tsv" > 2.log 2>&1 &

nohup python3 -u build_index.py \
--create-index false \
--index-name "ms-marco" \
--path-doc-embedding "../embedding_data/corpus/split03.pt" \
--path-doc "../data/collection.tsv" > 3.log 2>&1 &

nohup python3 -u build_index.py \
--create-index false \
--index-name "ms-marco" \
--path-doc-embedding "../embedding_data/corpus/split04.pt" \
--path-doc "../data/collection.tsv" > 4.log 2>&1 &

nohup python3 -u build_index.py \
--create-index false \
--index-name "ms-marco" \
--path-doc-embedding "../embedding_data/corpus/split05.pt" \
--path-doc "../data/collection.tsv" > 5.log 2>&1 &

nohup python3 -u build_index.py \
--create-index false \
--index-name "ms-marco" \
--path-doc-embedding "../embedding_data/corpus/split06.pt" \
--path-doc "../data/collection.tsv" > 6.log 2>&1 &

nohup python3 -u build_index.py \
--create-index false \
--index-name "ms-marco" \
--path-doc-embedding "../embedding_data/corpus/split07.pt" \
--path-doc "../data/collection.tsv" > 7.log 2>&1 &

nohup python3 -u build_index.py \
--create-index false \
--index-name "ms-marco" \
--path-doc-embedding "../embedding_data/corpus/split08.pt" \
--path-doc "../data/collection.tsv" > 8.log 2>&1 &

nohup python3 -u build_index.py \
--create-index false \
--index-name "ms-marco" \
--path-doc-embedding "../embedding_data/corpus/split09.pt" \
--path-doc "../data/collection.tsv" > 9.log 2>&1 &
```

### search

代码见[此](./Elasticsearch/search.py)

```bash
python3 -u search-queries.py \
--search-method "inverted-index" \
--inverted-index-key "doc" \
--path-query "../data/queries.dev.small.tsv" \
--path-search-result "./inverted-index-es.tsv" \
--path-latency-result "./latency-inverted-index-es.tsv"

python3 -u search-queries.py \
--search-method "knn" \
--knn-key "embedding" \
--path-query-embedding "../embedding_data/query/query_dev_small.pt" \
--path-search-result "./knn-es.tsv" \
--path-latency-result "./latency-knn-es.tsv"

python3 -u search-queries.py \
--search-method "combine" \
--inverted-index-key "doc" \
--path-query "../data/queries.dev.small.tsv" \
--knn-key "embedding" \
--path-query-embedding "../embedding_data/query/query_dev_small.pt" \
--path-search-result "./inverted-index-knn-es.tsv" \
--path-latency-result "./latency-inverted-index-knn-es.tsv" \
--knn-weight 800
```

### utils

```bash
#see index
curl -X GET https://localhost:9400/ms-marco/ --cacert http_ca.crt -u elastic

#see doc0
curl -X GET https://localhost:9400/ms-marco/_doc/0 --cacert http_ca.crt -u elastic
```

## SPANN

### install

首先安装`cmake`:

```bash
wget "https://github.com/Kitware/CMake/releases/download/v3.14.4/cmake-3.14.4-Linux-x86_64.tar.gz"
tar -zxvf cmake-3.14.4-Linux-x86_64.tar.gz

# Then add PATH to ~/.bashrc
export PATH=cmake-3.14.4-Linux-x86_64/bin:$PATH

# test installation
cmake --version

---------------------------------------------------------------------
cmake version 3.14.4

CMake suite maintained and supported by Kitware (kitware.com/cmake).
---------------------------------------------------------------------
```

然后安装`boost`:

```bash
wget https://boostorg.jfrog.io/artifactory/main/release/1.67.0/source/boost_1_67_0.tar.gz
tar -zxvf boost_1_67_0.tar.gz
cd boost_1_67_0
./bootstrap.sh --prefix=./boost #指定安装目录
./b2 install

# Then add BOOST_INCLUDEDIR and BOOST_LIBRARYDIR to ~/.bashrc
export BOOST_INCLUDEDIR=./boost/include:$BOOST_INCLUDEDIR
export BOOST_LIBRARYDIR=./boost/lib:$BOOST_LIBRARYDIR
```

Clone `sptag` and install:

```bash
git clone --recursive https://github.com/microsoft/SPTAG.git

cd SPTAG
mkdir build
cd build
cmake ..
make -j8
```

接下来和SPANN有关的代码都最好放在`Release`文件夹下运行。

关于SPANN的使用，可以参考[此](https://github.com/microsoft/SPTAG/blob/main/docs/Tutorial.ipynb)。

### build index

有几点需要注意:

- `InternalResultNum`和`SearchInternalResultNum`要**保持一致**, 一个是build的参数, 一个是search的参数, 是指search多少个posting.
- `PostingPageLimit`和`SearchPostingPageLimit`也需要**保持一致**, 原因同上, 是指一个posting最多有多少个4k page.
- `index.SetBuildParam("xxx", "xxx", "SearchSSDIndex")`貌似没有用, 所有的设置都要写成`index.SetBuildParam("xxx", "xxx", "BuildSSDIndex")`.

代码见[此](./SPANN/build-index.py), 使用下面的命令运行脚本(可能需要几个小时)。

```bash
python3 -u build-index.py 2>&1 > build-index.log &
```

构建完成后得到index的目录如下:

```bash
- HeadIndex
    - deletes.bin  
    - graph.bin  
    - indexloader.ini  
    - tree.bin  
    - vectors.bin
- indexloader.ini  
- SPTAGFullList.bin  
- SPTAGHeadVectorIDs.bin  
- SPTAGHeadVectors.bin
```

### search

`SPANN`搜索代码见[此](./SPANN/search.py), 使用下面的命令运行脚本。

```bash
python3 -u search.py 2>&1 > search.log &
```

`SPANN`+`Inverted Index`搜索代码见[此](./SPANN/hybrid-search.py), 基本思想是`SPANN`和`Elasticsearch`各搜索200个结果, 然后合并。使用下面的命令运行脚本。

```bash
python3 -u hybrid-search.py 2>&1 > hybrid-search.log &
```

## Evaluation

Accuracy部分的验证需要在**Windows Python3.9**环境下进行, 代码见[此](./Evaluation/eval_trec.py):

```bash
python -u eval_trec.py --trec_path "path-to-search-result" --qrels_path "./qrels.dev.small.tsv" --output_path "./results.tsv"
```

Latency部分代码见[此](./Evaluation/eval_latency.py):

```bash
python3 eval_latency.py
```

## Filter Search

为MS Marco数据随机生成**zipf**分布的`location`, 共有100个`location`, 每个`(str)location`对应一个`(int)tag`.

Zipf's law是美国语言学家Zipf发现的，他在1932年研究英文单词的出现频率时，发现如果把单词频率从高到低的次序排列，每个单词出现频率 $P(r)$ 和它的频率排名 $r$ 存在简单反比关系:

$$P(r) = \frac{C}{r^\alpha}$$

这个定理说明只有少数单词被经常使用, 这一定律也在互联网内容访问中成立。

我们生成的zipf分布中 $C=0.1928$, $\alpha=1$, 概率密度图如下:

<div align=center>
<img src="./FilterSearch/zipf.jpg" width=80%/>
</div>
</br>

在`FilterSearch`文件夹中:

- `gen_filter.py`用于生成`location`和`tag`.
- `gen_gt.py`用于生成ground truth.
- `spann_filter.py`每次**double** SPANN返回的结果直到filter后剩余结果超过100个，该结果作为baseline.
- `eval.py`用于计算filter search的`MRR`和`Recall`.