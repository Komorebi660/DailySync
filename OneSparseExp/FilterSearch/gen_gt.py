import pickle
import csv
import numpy as np

print("Load Embedding Begin !!!")
corpus = []
for i in range(10):
    with open("../embedding_data/corpus/split0%d.pt" % i, "rb") as f:
        embedding, _ = pickle.load(f)
        corpus.append(embedding)
    f.close()
    print("finish %d" % i)
corpus = np.vstack(corpus)
print("Load Embedding End !!!")

print("Load Location Begin !!!")
location = []
with open("./passage_filter.tsv", "r", encoding="utf8") as f:
    tsvreader_query = csv.reader(f, delimiter="\t")
    for [tag, country] in tsvreader_query:
        location.append(tag)
    f.close()
location = np.array(location)
print("Load Location End !!!")

with open("../embedding_data/query/query_dev_small.pt", 'rb') as f_embedding, \
        open("./query_filter.tsv", "r", encoding="utf8") as f_filter, \
        open("./gt.tsv", "w", encoding="utf8") as f_result:
    tsvreader_query = csv.reader(f_filter, delimiter="\t")
    query_embeddings, _ = pickle.load(f_embedding)
    idx = 0
    for [qid, loc, _] in tsvreader_query:
        query_embedding = np.array(query_embeddings[idx].tolist(), dtype=np.float32)
        # find the index of loc in location
        index = np.where(location == loc)[0]
        # filter passages
        candidate_corpus = corpus[index]
        # calculate scores
        score_list = np.linalg.norm(candidate_corpus-query_embedding, axis=1)**2
        # sort
        sorted_index = np.argsort(score_list)
        for i in range(100):
            # docid start from 0
            f_result.write(f"{qid}\t{index[sorted_index[i]]}\t{i+1}\t{100.0-score_list[sorted_index[i]]}\n")
        idx += 1
        if idx % 100 == 0:
            f_result.flush()
            print(f"{idx} queries searched...")
    print(f"{idx} queries searched.")
