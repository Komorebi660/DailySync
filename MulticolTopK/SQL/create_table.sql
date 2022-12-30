create database test_db;
\c test_db;
create extension if not exists vector;
create extension if not exists vectordb;
insert into model values ('word2vec', 'word2vec', '0');

create table d_table(id int PRIMARY KEY,image_embedding double precision[1024],text_embedding double precision[1024],content text,ingre_count int,instr_step int,bm25 jsonb);
copy d_table from '/vectordb/data.tsv' DELIMITER E'\t' csv quote e'\x01';

create index hnsw_index1 on d_table using hnsw(image_embedding hnsw_vector_inner_product_ops) with (dimension='1024', distmethod=inner_product);
create index hnsw_index2 on d_table using hnsw(text_embedding hnsw_vector_inner_product_ops) WITH (dimension='1024', distmethod=inner_product);

\d d_table