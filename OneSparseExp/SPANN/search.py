import sys
import pickle
import time
import numpy as np

sys.path.append('/SPTAG/Release/')
import SPTAG

index = SPTAG.AnnIndex.Load('msmarco')

with open("../../embedding_data/query/query_dev_small.pt", 'rb') as f, \
    open("./spann_qrels.tsv", 'w', encoding="utf8") as out, \
    open("./spann_latency.tsv", 'w', encoding="utf8") as out_latency:
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
