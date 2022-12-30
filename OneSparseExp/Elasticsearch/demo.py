import json
import sys
import argparse
import requests

requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

user = 'elastic'
password = ''
cert = 'http_ca.crt'


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def create_index(index_name):
    url = f"https://localhost:9400/{index_name}"
    headers = {
        'Content-Type': 'application/json'
    }
    payload = {
        "mappings": {
            "properties": {
                "embedding": {
                    "type": "dense_vector",
                    "dims": 3,
                    "index": True,
                    "index_options": {
                        "type": "hnsw",
                        "m": 16,
                        "ef_construction": 100
                    },
                    "similarity": "cosine"  # l2_norm, dot_product, cosine
                },
                "doc": {
                    "type": "text",
                }
            }
        }
    }
    #response = requests.request("DELETE", url, headers=headers, verify=cert, auth=(user, password))
    response = requests.request("PUT", url, headers=headers, data=json.dumps(payload), verify=cert, auth=(user, password))
    if not response.ok:
        print(response.text)
        sys.exit(1)


def add_data_to_index(index_name, docid, embedding, doc):
    headers = {
        'Content-Type': 'application/json'
    }

    url = f"https://localhost:9400/{index_name}/_doc/{docid}"
    payload = {
        "embedding": embedding,
        "doc": doc
    }
    response = requests.request("PUT", url, headers=headers, data=json.dumps(payload), verify=cert, auth=(user, password))
    if not response.ok:
        print(response.text)
        sys.exit(1)


def search_with_inverted_index(index_name, key, query):
    url = f"https://localhost:9400/{index_name}/_search"
    headers = {
        'Content-Type': 'application/json'
    }
    payload = {
        "size": 10,  # number of results
        "query": {
            "match": {
                key: query,
            },
        },
    }
    response = requests.request("GET", url, headers=headers, data=json.dumps(payload), verify=cert, auth=(user, password))
    res = json.loads(response.text)
    print(res)


def search_with_knn(index_name, key, query_embedding):
    url = f"https://localhost:9400/{index_name}/_search"
    headers = {
        'Content-Type': 'application/json'
    }
    payload = {
        "knn": {
            "field": key,
            "query_vector": query_embedding,
            "k": 2,  # number of results
            "num_candidates": 100
        }
    }
    response = requests.request("GET", url, headers=headers, data=json.dumps(payload), verify=cert, auth=(user, password))
    res = json.loads(response.text)
    print(res)


# https://www.elastic.co/guide/en/elasticsearch/reference/8.4/knn-search.html#approximate-knn
def search_with_inverted_index_and_knn(index_name, inverted_index_key, knn_key, knn_weight, query, query_embedding):
    url = f"https://localhost:9400/{index_name}/_search"
    headers = {
        'Content-Type': 'application/json'
    }
    payload = {
        "size": 10,
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
            "k": 10,
            "num_candidates": 100,
            "boost": knn_weight
        },
    }
    response = requests.request("GET", url, headers=headers, data=json.dumps(payload), verify=cert, auth=(user, password))
    res = json.loads(response.text)
    print(res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--create-index', type=str2bool, default=False,
                        help='whether to create index')
    parser.add_argument('--index-name', type=str, default="ms-index",
                        help='index name')

    args = parser.parse_args()

    if args.create_index:
        create_index(args.index_name)

        add_data_to_index(args.index_name, "0", [0.5, 10, 6], "my test")
        add_data_to_index(args.index_name, "1", [-0.5, 10, 10], "your test")

    search_with_inverted_index(args.index_name, "doc", "test")
    search_with_knn(args.index_name, "embedding", [0.5, 10, 6])
    search_with_inverted_index_and_knn(args.index_name, "doc", "embedding", 50, "test", [-0.5, 10, 10])
