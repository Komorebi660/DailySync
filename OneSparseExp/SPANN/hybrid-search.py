import SPTAG
import requests
import pickle
import csv
import time
import json
import sys
import numpy as np

sys.path.append('/SPTAG/Release/')

index = SPTAG.AnnIndex.Load('msmarco')

index_name = "ms-marco"
user = 'elastic'
password = ''
cert = 'http_ca.crt'


def search_with_inverted_index(query):
    """return a tuple(docid, rank, score)
    """
    url = f"https://localhost:9400/{index_name}/_search"
    headers = {
        'Content-Type': 'application/json'
    }
    payload = {
        "size": 200,  # number of results
        "query": {
            "match": {
                "doc": query,
            },
        },
    }
    response = requests.request("GET", url, headers=headers, data=json.dumps(payload), verify=cert, auth=(user, password))
    res = json.loads(response.text)

    #record["_source"]["embedding"]
    return [(record["_id"], idx+1, record["_score"]) for idx, record in enumerate(res["hits"]["hits"])]


def search_with_spann(embedding):
    """return a tuple(docid, rank, score)
    """
    embedding = np.hstack((0.0, embedding)).astype(np.float32)
    result = index.Search(embedding, 200)
    return [(result[0][i], i+1, 100.0-result[1][i]) for i in range(200)]


def merge(query_path, query_embedding_path, out_path, out_latency_path, knn_weight):
    with open(query_path, "r", encoding="utf8") as f_query, \
            open(query_embedding_path, 'rb') as f_query_embedding, \
            open(out_path, 'w', encoding="utf8") as out, \
            open(out_latency_path, 'w', encoding="utf8") as out_latency:

        tsvreader_query = csv.reader(f_query, delimiter="\t")
        query_embeddings, _ = pickle.load(f_query_embedding)
        idx = 0
        for [qid, query] in tsvreader_query:
            query_embedding = np.array(query_embeddings[idx].tolist(), dtype=np.float32)
            latency = 0.0
            start = time.time()
            #store candidates
            candidates = {}  # docid: [spann_score, ivf_score, merged_score]

            #search with spann and return 200 results
            result1 = search_with_spann(query_embedding)
            for (docid, rank, score) in result1:
                #top200 add to candidates
                candidates[int(docid)] = [score, 0.0, knn_weight * score]  # use spann_score as ivf_score

            #search with inverted index and return 200 results
            result2 = search_with_inverted_index(query)
            for (docid, rank, score) in result2:
                # merge top200 to existed candidates
                if int(docid) in candidates:
                    candidates[int(docid)][1] = score
                    candidates[int(docid)][2] = score + knn_weight * candidates[int(docid)][0]
                else:
                    candidates[int(docid)] = [0.0, score, score]

            # sort [docid, [spann_score, ivf_score, merged_score]]
            candidates = sorted(candidates.items(), key=lambda x: x[1][2], reverse=True)
            latency += time.time() - start
            for i in range(100):
                out.write(f"{qid} 0 {candidates[i][0]} {i+1} {candidates[i][1][2]} IndriQueryLikelihood\n")
            out_latency.write(f"{qid}\t{latency}\n")

            idx += 1
            if idx % 100 == 0:
                out.flush()
                out_latency.flush()
                print(f"{idx} queries searched...")
        print(f"{idx} queries searched.")


if __name__ == "__main__":
    query_path = "../../data/queries.dev.small.tsv"
    query_embedding_path = "../../embedding_data/query/query_dev_small.pt"
    out_path = "./inverted_index_spann_qrels.tsv"
    out_latency_path = "./inverted_index_spann_latency.tsv"
    knn_weight = 10.0
    merge(query_path, query_embedding_path, out_path, out_latency_path, knn_weight)
