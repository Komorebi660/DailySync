import csv
import json
import argparse
import requests
import pickle

requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

user = 'elastic'
password = ''
cert = 'http_ca.crt'

index_name = "ms-marco"

s = requests.session()
s.keep_alive = False  # avoid too many connections

def search_with_inverted_index_and_knn(inverted_index_key, knn_key, knn_weight, query, query_embedding):
    """return a tuple(docid, rank, score, latency)
    """
    url = f"https://localhost:9400/{index_name}/_search"
    headers = {
        'Content-Type': 'application/json'
    }
    payload = {
        "size": 100,
        "query": {
            "match": {
                inverted_index_key: {
                    "query": query,
                    "boost": 1
                }
            }
        },
        "knn": {
            "field": knn_key,
            "query_vector": query_embedding,
            "k": 100,
            "num_candidates": 100,
            "boost": knn_weight
        },
    }
    response = s.request("GET", url, headers=headers, data=json.dumps(payload), verify=cert, auth=(user, password))
    res = json.loads(response.text)

    latency = response.elapsed.total_seconds()

    return [(record["_source"]["docid"], idx+1, record["_score"]) for idx, record in enumerate(res["hits"]["hits"])], latency


def search_with_inverted_index(key, query):
    """return a tuple(docid, rank, score, latency)
    """
    url = f"https://localhost:9400/{index_name}/_search"
    headers = {
        'Content-Type': 'application/json'
    }
    payload = {
        "size": 100,  # number of results
        "query": {
            "match": {
                key: query,
            },
        },
    }
    response = s.request("GET", url, headers=headers, data=json.dumps(payload), verify=cert, auth=(user, password))
    res = json.loads(response.text)

    latency = response.elapsed.total_seconds()

    return [(record["_source"]["docid"], idx+1, record["_score"]) for idx, record in enumerate(res["hits"]["hits"])], latency


def search_with_knn(key, query_embedding):
    """return a tuple(docid, rank, score, latency)
    """
    url = f"https://localhost:9400/{index_name}/_search"
    headers = {
        'Content-Type': 'application/json'
    }
    payload = {
        "size": 100,
        "knn": {
            "field": key,
            "query_vector": query_embedding,
            "k": 100,  # number of results
            "num_candidates": 100
        }
    }
    response = s.request("GET", url, headers=headers, data=json.dumps(payload), verify=cert, auth=(user, password))
    res = json.loads(response.text)

    latency = response.elapsed.total_seconds()
    return [(record["_source"]["docid"], idx+1, record["_score"]) for idx, record in enumerate(res["hits"]["hits"])], latency


def search_queries_with_inverted_index(query_path, key, qrels_path, latency_path, queries=-1, print_frequency=100):
    with open(query_path, "r", encoding="utf8") as f, \
            open(qrels_path, 'w', encoding="utf8") as out, \
            open(latency_path, 'w', encoding="utf8") as out_latency:
        tsvreader = csv.reader(f, delimiter="\t")
        idx = 0
        for [qid, query] in tsvreader:
            result, latency = search_with_inverted_index(key, query)
            # save search result in TREC format `qid 0 pid rank score IndriQueryLikelihood & format `qid latency`.
            for (docid, rank, score) in result:
                out.write(f"{qid} 0 {docid} {rank} {score} IndriQueryLikelihood\n")
            out_latency.write(f"{qid}\t{latency}\n")

            idx += 1
            if idx % print_frequency == 0:
                out.flush()
                out_latency.flush()
                print(f"{idx} queries searched...")
            if queries != -1 and idx >= queries:
                break
        print(f"{idx} queries searched.")


def search_queries_with_knn(query_embedding_path, key, qrels_path, latency_path, queries=-1, print_frequency=100):
    with open(query_embedding_path, 'rb') as f, \
            open(qrels_path, 'w', encoding="utf8") as out, \
            open(latency_path, 'w', encoding="utf8") as out_latency:
        embeddings, ids = pickle.load(f)
        for idx in range(len(ids)):
            qid = int(ids[idx])
            embedding = embeddings[idx].tolist()
            result, latency = search_with_knn(key, embedding)
            # save search result in TREC format `qid 0 pid rank score IndriQueryLikelihood & format `qid latency`.
            for (docid, rank, score) in result:
                out.write(f"{qid} 0 {docid} {rank} {score} IndriQueryLikelihood\n")
            out_latency.write(f"{qid}\t{latency}\n")

            if idx % print_frequency == 0:
                out.flush()
                out_latency.flush()
                print(f"{idx} queries searched...")
            if queries != -1 and idx >= queries-1:
                break
        print(f"{len(ids)} queries searched.")


def search_queries_with_inverted_index_and_knn(
        query_path, inverted_index_key, query_embedding_path, knn_key, knn_weight, qrels_path, latency_path, queries=-1, print_frequency=100):
    with open(query_path, "r", encoding="utf8") as f_query, \
            open(query_embedding_path, 'rb') as f_query_embedding, \
            open(qrels_path, 'w', encoding="utf8") as out, \
            open(latency_path, 'w', encoding="utf8") as out_latency:
        tsvreader_query = csv.reader(f_query, delimiter="\t")
        query_embeddings, _ = pickle.load(f_query_embedding)
        idx = 0
        #embedding and original query data must be aligned
        for [qid, query] in tsvreader_query:
            query_embedding = query_embeddings[idx].tolist()
            result, latency = search_with_inverted_index_and_knn(inverted_index_key, knn_key, knn_weight, query, query_embedding)
            # save search result in TREC format `qid 0 pid rank score IndriQueryLikelihood & format `qid latency`.
            for (docid, rank, score) in result:
                out.write(f"{qid} 0 {docid} {rank} {score} IndriQueryLikelihood\n")
            out_latency.write(f"{qid}\t{latency}\n")

            idx += 1
            if idx % print_frequency == 0:
                out.flush()
                out_latency.flush()
                print(f"{idx} queries searched...")
            if queries != -1 and idx >= queries:
                break
        print(f"{idx} queries searched.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-query', type=str, default='../data/queries.dev.small.tsv',
                        help='path to query')
    parser.add_argument('--path-query-embedding', type=str,
                        default='../embedding_data/query/query_dev_small.pt',
                        help='path to query embedding result')
    parser.add_argument('--search-method', type=str, default='combine',
                        help='inverted-index, knn, combine')
    parser.add_argument('--inverted-index-key', type=str, default='doc',
                        help='doc')
    parser.add_argument('--knn-key', type=str, default='embedding',
                        help='embedding')
    parser.add_argument('--knn-weight', type=int, default=40,
                        help='wight of knn in combine search')
    parser.add_argument('--path-search-result', type=str, default="./es-qrels.tsv",
                        help='path to save search result')
    parser.add_argument('--path-latency-result', type=str, default="./es-latency.tsv",
                        help='path to save latency result')
    parser.add_argument('--queries-to-search', type=int, default=-1,
                        help='number of queries to search')
    parser.add_argument('--log-frequency', type=int, default=100,
                        help='log frequency')

    args = parser.parse_args()

    if args.search_method == 'inverted-index':
        search_queries_with_inverted_index(
            args.path_query, args.inverted_index_key,
            args.path_search_result, args.path_latency_result,
            args.queries_to_search, args.log_frequency)
    elif args.search_method == 'knn':
        search_queries_with_knn(
            args.path_query_embedding, args.knn_key,
            args.path_search_result, args.path_latency_result,
            args.queries_to_search, args.log_frequency)
    elif args.search_method == 'combine':
        search_queries_with_inverted_index_and_knn(
            args.path_query, args.inverted_index_key,
            args.path_query_embedding, args.knn_key,
            args.knn_weight,
            args.path_search_result, args.path_latency_result,
            args.queries_to_search, args.log_frequency)
