# Multi-Column TopK Experiment

- [Multi-Column TopK Experiment](#multi-column-topk-experiment)
  - [Run Scripts](#run-scripts)
  - [Function Defination](#function-defination)
  - [Demo SQL](#demo-sql)
  - [Utils](#utils)

## Run Scripts

Clone the repository:

```bash
git clone -b yaoqi/multicol --recursive qiazh@vs-ssh.visualstudio.com:v3/qiazh/vectordb/vectordb
cd vectordb/tests_cyq
```

Build & run docker:

```bash
docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t vectordb_cyq -f ../Dockerfile.dev ..

docker run --name=vectordb_cyq -e PGPASSWORD=vectordb -e PGUSERNAME=vectordb -e PGDATABASE=vectordb -v `pwd`/..:/vectordb  vectordb_cyq &
```

Get Data:

```bash
scp qianxi@10.190.172.91:/data/qianxi/VBASE/bm25/data/data.tsv ../
```

Create Table:

```bash
docker exec -it vectordb_cyq psql -U vectordb -f /vectordb/tests_cyq/create_table.sql
```

| id | image_embedding | text_embedding | ... |
| :-----: | :-----: | :-----: | :-----: |
| int(_PK_) | double[1024] | double[1024] | ... |

Run Query:

```bash
#enter docker
docker exec -it vectordb_cyq bash
#go to vectordb
cd vectordb

# `-f` means run query from file
# run without define dev (use to test performance)
psql -U vectordb -f ./tests_cyq/query.sql > result.txt 2>&1
psql -U vectordb -f ./tests_cyq/query_gt.sql > gt.txt 2>&1

# run with define dev (use to get traversal trends)
psql -U vectordb -f ./tests_cyq/query.sql 2> log.txt
```

Update vectordb (rebuild):

```bash
#before first run, need to run `mkdir build` in docker
docker exec -it -u 0 vectordb_cyq bash -c "cd /vectordb/build && cmake .. && make -j 8 && make install"
```

代码不同的define组合对应不同的代码版本, 每次改动需要rebuild:

| define        | code        | print internal results or not |
| :-----------: | :---------: | :---: |
|               | round-robin | no  |
| dev           | round-robin | yes |
| opt, one      | greedy      | no  |
| opt, one, dev | greedy      | yes |
| opt           | optimized   | no  |
| opt, dev      | optimized   | yes |

## Function Defination

```
multicol_topk(table_name text, k integer, term_cond integer, attr_exp text, filter_exp text, rank_exp text, VARIADIC arr text[])
```

- `table_name`: name of table which runs the query (`from xxx`)
- `k`: number of returned results (`limit xxx`)
- `term_cond`: termination condition, determines how many times the priority queue is not changed before exited
- `attr_exp`: what attributes needed to be returned (`select xxx`)
- `filter_exp`: filter condition (`where xxx`)
- `rank_exp`: rank expression (`order by xxx`), if empty, use simple summation of all `arr`
- `arr`: array of expressions, each expression is an element of ranking function, e.g. `image_embedding <*> ARRAY[0.1, ...]`

## Demo SQL

```sql
\c test_db;

select id, (image_embedding <*> (select image_embedding from d_table where id=0)) + (text_embedding <*> (select text_embedding from d_table where id=0)) as tag from d_table order by image_embedding <*> (select image_embedding from d_table where id=0) limit 10;

select id, (image_embedding <*> (select image_embedding from d_table where id=0)) + (text_embedding <*> (select text_embedding from d_table where id=0)) as tag from d_table order by text_embedding <*> (select text_embedding from d_table where id=0) limit 10;

select id, (image_embedding <*> (select image_embedding from d_table where id=0)) + (text_embedding <*> (select text_embedding from d_table where id=0)) as tag from d_table order by tag limit 10;

select multicol_topk('d_table', 10, 200, 'id', '', '', 'image_embedding <*> (select image_embedding from d_table where id=0)', 'text_embedding <*> (select text_embedding from d_table where id=0)');
```

## Utils

Enter docker: 

```bash
docker exec -it vectordb_cyq bash
```

Enter docker & open psgl console:

```bash
docker exec -it vectordb_cyq bash -c "psql -U vectordb"
```

Run file:

```bash
docker exec -it vectordb_cyq bash
psql -U vectordb -f xxx.sql
```

PostgresSQL常用操作如下:

```bash
#进入控制台
psql -U vectordb

#退出
\q

#查看所有数据库
\l

#切换数据库
\c xxx

#列出当前数据库的所有表格
\d
```

更多命令可以点击点[这里](https://mozillazg.com/2014/06/hello-postgresql.html)。