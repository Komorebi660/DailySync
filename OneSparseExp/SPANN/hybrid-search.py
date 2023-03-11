import requests
import pickle
import csv
import time
import json
import numpy as np
import argparse
import SPTAG

index = SPTAG.AnnIndex.Load('msmarco')

index_name = "ms-marco"
user = 'elastic'
password = ''
cert = 'path-to-http_ca.crt'

s = requests.session()
s.keep_alive = False

def search_with_inverted_index(query, k):
    """return a tuple(docid, rank, score)
    """
    url = f"https://localhost:9400/{index_name}/_search"
    headers = {
        'Content-Type': 'application/json'
    }
    payload = {
        "size": k,  # number of results
        "query": {
            "match": {
                "doc": query,
            },
        },
        "_source" : False,
    }
    response = s.request("GET", url, headers=headers, data=json.dumps(payload), verify=cert, auth=(user, password))
    res = json.loads(response.text)

    #latency = response.elapsed.total_seconds()

    #record["_source"]["embedding"]
    return [(record["_id"], idx+1, record["_score"]) for idx, record in enumerate(res["hits"]["hits"])], res['took']/1000.0


def search_with_spann(embedding, k):
    """return a tuple(docid, rank, score)
    """
    start = time.time()
    result = index.Search(embedding, k)
    latency = time.time() - start
    return [(result[0][i], i+1, 1.0/(1.0+float(result[1][i]))) for i in range(len(result[1]))], latency


def merge(query_path, query_embedding_path, out_path, out_latency_path, k1, k2, knn_weight):
    with open(query_path, "r", encoding="utf8") as f_query, \
            open(query_embedding_path, 'rb') as f_query_embedding, \
            open(out_path, 'w', encoding="utf8") as out, \
            open(out_latency_path, 'w', encoding="utf8") as out_latency:
        
        tsvreader_query = csv.reader(f_query, delimiter="\t")
        query_embeddings, _ = pickle.load(f_query_embedding)
        idx = 0
        _latency, spann_latency, iv_latency, merge_latency = 0.0, 0.0, 0.0, 0.0
        for [qid, query] in tsvreader_query:
            query_embedding = np.array(query_embeddings[idx].tolist(), dtype=np.float32)
            #store candidates
            candidates = {}  # docid: [spann_score, ivf_score, merged_score]

            #search with spann and return k1 results
            result1, time1 = search_with_spann(query_embedding, k1)
            for (docid, rank, score) in result1:
                candidates[int(docid)] = [score, 0.0, score]

            #search with inverted index and return k2 results
            result2, time2 = search_with_inverted_index(query, k2)

            start = time.time()
            for (docid, rank, score) in result2:
                # merge k2 to existed candidates
                if int(docid) in candidates:
                    candidates[int(docid)][1] = score
                    candidates[int(docid)][2] = score + knn_weight * candidates[int(docid)][0]
                #else:
                #    candidates[int(docid)] = [0.0, score, score]
            
            for did in list(candidates.keys()):
                if candidates[did][1] == 0.0:
                    del candidates[did]

            # sort [docid, [spann_score, ivf_score, merged_score]]
            candidates = sorted(candidates.items(), key=lambda x: x[1][2], reverse=True)
            time3 = time.time() - start
            
            latency = max(time1, time2) + time3
            spann_latency += time1
            iv_latency += time2
            merge_latency += time3
            _latency += (time1 + time2 + time3)

            num = 100 if len(candidates)>100 else len(candidates)
            for i in range(num):
                out.write(f"{qid}\t{candidates[i][0]}\t{i+1}\t{candidates[i][1][2]}\n")
            out_latency.write(f"{qid}\t{latency}\n")

            idx += 1
            if idx % 100 == 0:
                out.flush()
                out_latency.flush()
                print(f"{idx} queries searched...")
        
        print(f"{idx} queries searched.")
        print(f"QPS: {idx/_latency}")
        print(f"Avg SPANN Latency: {spann_latency/idx}")
        print(f"Avg Inverted Index Latency: {iv_latency/idx}")
        print(f"Avg Merge Latency: {merge_latency/idx}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--query-path', type=str, default="../data/queries.dev.small.tsv",
                        help='path to query')
    parser.add_argument('--query-embedding-path', type=str, default="../embedding_data/query/query_dev_small.pt",
                        help='path to query embeddings')
    parser.add_argument('--search-result-path-prefix', type=str, default="./inverted_index_spann_qrels_",
                        help='path to save search result')
    parser.add_argument('--latency-result-path-prefix', type=str, default="./inverted_index_spann_latency_",
                        help='path to save latency result')
    parser.add_argument('--k1', type=int, default=5000,
                        help='results retrieved from SPANN')
    parser.add_argument('--k2', type=int, default=10000,
                        help='results retrieved from inverted index')                 

    args = parser.parse_args()

    result_path = args.search_result_path_prefix + str(args.k1) + "_" + str(args.k2) + ".tsv"
    latency_path = args.latency_result_path_prefix + str(args.k1) + "_" + str(args.k2) + ".tsv"

    merge(args.query_path, args.query_embedding_path, result_path, latency_path, \
          args.k1, args.k2, 15000.0)
