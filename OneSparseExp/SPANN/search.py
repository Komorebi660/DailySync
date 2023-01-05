import sys
import pickle
import time
import numpy as np
import argparse

sys.path.append('/SPTAG/Release/')
import SPTAG

def search(query_path, result_path, latency_path):
    index = SPTAG.AnnIndex.Load('msmarco')

    with open(query_path, 'rb') as f, \
        open(result_path, 'w', encoding="utf8") as out, \
        open(latency_path, 'w', encoding="utf8") as out_latency:
            embeddings, ids = pickle.load(f)
            for idx in range(len(ids)):
                qid = int(ids[idx])
                embedding = np.array(embeddings[idx].tolist(), dtype=np.float32)
                start = time.time()
                result = index.Search(embedding, 100)
                end = time.time()
                latency = end - start
                # save search result in TREC format `qid 0 pid rank score IndriQueryLikelihood & format `qid latency`.
                #print(result)
                for i in range(100):
                    out.write(f"{qid} 0 {result[0][i]} {i+1} {100-result[1][i]} IndriQueryLikelihood\n")
                out_latency.write(f"{qid}\t{latency}\n")

                if idx % 100 == 0:
                    out.flush()
                    out_latency.flush()
                    print(f"{idx} queries searched...")
            print(f"{len(ids)} queries searched.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--query-path', type=str, default="../embedding_data/query/query_dev_small.pt",
                        help='path to query embeddings')
    parser.add_argument('--search-result-path', type=str, default="./spann_qrels.tsv",
                        help='path to save search result')
    parser.add_argument('--latency-result-path', type=str, default="./spann_latency.tsv",
                        help='path to save latency result')

    args = parser.parse_args()

    search(args.query_path, args.search_result_path, args.latency_result_path)
