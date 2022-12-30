import pickle
import csv
import time
import sys
import numpy as np

sys.path.append('/SPTAG/Release/')
import SPTAG

index = SPTAG.AnnIndex.Load('msmarco')


def load_passage_filter(path):
    tags = []
    with open(path, "r", encoding="utf8") as f:
        tsvreader_query = csv.reader(f, delimiter="\t")
        for [tag, _] in tsvreader_query:
            tags.append(tag)
        f.close()
    return np.array(tags)


def search_with_spann(embedding, k):
    """return a tuple(docid, rank, score)
    """
    embedding = np.hstack((0.0, embedding)).astype(np.float32)
    result = index.Search(embedding, k)
    return [(result[0][i], 100.0-result[1][i]) for i in range(k)]


def search(passage_filter_path, query_filter_path, query_embedding_path, out_path, out_latency_path):
    passage_filter = load_passage_filter(passage_filter_path)

    with open(query_filter_path, "r", encoding="utf8") as f_query_filter, \
            open(query_embedding_path, 'rb') as f_query_embedding, \
            open(out_path, 'w', encoding="utf8") as out, \
            open(out_latency_path, 'w', encoding="utf8") as out_latency:

        tsvreader_query = csv.reader(f_query_filter, delimiter="\t")
        query_embeddings, _ = pickle.load(f_query_embedding)
        idx = 0
        total_k = 0
        for [qid, tag, _] in tsvreader_query:
            query_embedding = np.array(query_embeddings[idx].tolist(), dtype=np.float32)
            latency = 0.0
            start = time.time()

            k = 100
            candidates = []  # [docid, score]
            #enlarge k until get enough candidates
            while len(candidates) < 100:
                k = k*2  # double k
                candidates = []  # clear
                #search with spann and return k results
                result = search_with_spann(query_embedding, k)
                for (docid, score) in result:
                    #filter
                    if passage_filter[int(docid)] != tag:
                        continue
                    candidates.append([int(docid), float(score)])

            latency += time.time() - start
            total_k += k
            for i in range(100):
                out.write(f"{qid}\t{candidates[i][0]}\t{i+1}\t{candidates[i][1]}\n")
            out_latency.write(f"{qid}\t{latency}\n")

            idx += 1
            if idx % 100 == 0:
                out.flush()
                out_latency.flush()
                print(f"{idx} queries searched, average k is {total_k/idx}.")
        print(f"{idx} queries searched, average k is {total_k/idx}.")


if __name__ == "__main__":
    passage_filter_path = "passage_filter.tsv"
    query_filter_path = "query_filter.tsv"
    query_embedding_path = "/embedding_data/query/query_dev_small.pt"
    out_path = "spann_filter_qrels.tsv"
    out_latency_path = "spann_filter_latency.tsv"
    search(passage_filter_path, query_filter_path, query_embedding_path, out_path, out_latency_path)
