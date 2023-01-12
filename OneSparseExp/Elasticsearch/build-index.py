import csv
import json
import sys
import argparse
import requests
import pickle

requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

user = 'elastic'
password = ''
cert = 'http_ca.crt'

s = requests.session()
s.keep_alive = False  # avoid too many connections

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
                    "dims": 769,
                    "index": True,
                    "index_options": {
                        "type": "hnsw",
                        "m": 16,
                        "ef_construction": 100
                    },
                    "similarity": "dot_product"  # l2_norm, dot_product, cosine
                },
                "doc": {
                    "type": "text",
                }
            }
        }
    }
    response = s.request("PUT", url, headers=headers, data=json.dumps(payload), verify=cert, auth=(user, password))
    print(response.text)


def add_data_to_index(index_name, docid, doc, embedding):
    headers = {
        'Content-Type': 'application/json'
    }
    url = f"https://localhost:9400/{index_name}/_doc/{docid}"
    payload = {
        "docid": docid,
        "doc": doc,
        "embedding": embedding,
    }

    response = s.request("PUT", url, headers=headers, data=json.dumps(payload), verify=cert, auth=(user, password))
    if not response.ok:
        print(response.text)
        sys.exit(1)


def update_doc_to_index(index_name, docid, doc):
    headers = {
        'Content-Type': 'application/json'
    }
    url = f"https://localhost:9400/{index_name}/_update/{docid}"
    payload = {
        "doc": {
            "doc": doc
        }
    }

    response = s.request("POST", url, headers=headers, data=json.dumps(payload), verify=cert, auth=(user, password))
    if not response.ok:
        print(response.text)
        sys.exit(1)


def update_emdedding_to_index(index_name, docid, embedding):
    headers = {
        'Content-Type': 'application/json'
    }
    url = f"https://localhost:9400/{index_name}/_update/{docid}"
    payload = {
        "doc": {
            "embedding": embedding
        }
    }

    response = s.request("POST", url, headers=headers, data=json.dumps(payload), verify=cert, auth=(user, password))
    if not response.ok:
        print(response.text)
        sys.exit(1)


def save_document(doc_path, embedding_path, index_name, print_frequency=1000):
    with open(embedding_path, 'rb') as f_embedding, \
            open(doc_path, "r", encoding="utf8") as f_doc:
        embeddings, ids = pickle.load(f_embedding)
        # embeddings are split into multiple files, so we need to know the starting line number
        start_line = int(ids[0])
        datasize = len(ids)
        tsvreader = csv.reader(f_doc, delimiter="\t")
        idx = 0
        for [pid, doc] in tsvreader:
            #go to start line
            if int(pid) < start_line:
                continue
            add_data_to_index(index_name, int(pid), doc, embeddings[idx].tolist())

            idx += 1
            if int(pid) % print_frequency == 0:
                print(f"{pid} passages saved...")
            if idx >= datasize:
                break
        print(f"{ids[-1]} passages saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--create-index', type=str2bool, default=True,
                        help='whether to create index')
    parser.add_argument('--index-name', type=str, default="ms-marco",
                        help='index name')
    parser.add_argument('--path-doc', type=str, default="../embedding_data/corpus/split00.pt",
                        help='path to docs/passages')
    parser.add_argument('--path-doc-embedding', type=str, default="../data/collection.tsv",
                        help='path to doc embedding result')

    args = parser.parse_args()

    if args.create_index:
        create_index(args.index_name)

    save_document(args.path_doc, args.path_doc_embedding, args.index_name)
