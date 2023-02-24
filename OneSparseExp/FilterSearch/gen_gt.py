import pickle
import csv
import numpy as np
import argparse

def load_passages(path):
    print("Load Embedding Begin !!!")
    corpus = []
    for i in range(10):
        with open(path + "%d.pt" % i, "rb") as f:
            embedding, _ = pickle.load(f)
            corpus.append(embedding)
        print("finish %d" % i)
    corpus = np.vstack(corpus)
    print(corpus.shape)
    print("Load Embedding End !!!")
    return corpus

def load_passage_filter(path):
    print("Load Location Begin !!!")
    location = []
    with open(path, "r", encoding="utf8") as f:
        tsvreader_query = csv.reader(f, delimiter="\t")
        for [tag, country] in tsvreader_query:
            location.append(tag)
    location = np.array(location)
    print("Load Location End !!!")
    return location

def gen(passage_path, passage_filter_path, query_path, query_filter_path, result_path):
    corpus = load_passages(passage_path)
    location = load_passage_filter(passage_filter_path)
    with open(query_path, 'rb') as f_embedding, \
         open(query_filter_path, "r", encoding="utf8") as f_filter, \
         open(result_path, "w", encoding="utf8") as f_result:
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
                f_result.write(f"{qid}\t{index[sorted_index[i]]}\t{i+1}\t{1.0/(1.0+float(score_list[sorted_index[i]]))}\n")
            idx += 1
            if idx % 100 == 0:
                f_result.flush()
                print(f"{idx} queries searched...")
        print(f"{idx} queries searched.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--passage-path-prefix', type=str, default="../embedding_data/corpus/split0",
                        help='prefix of path to passage embeddings')
    parser.add_argument('--passage-filter-path', type=str, default="passage_filter.tsv",
                        help='path to filter data of passages')
    parser.add_argument('--query-path', type=str, default="../embedding_data/query/query_dev_small.pt",
                        help='path to query embeddings')
    parser.add_argument('--query-filter-path', type=str, default="query_filter.tsv",
                        help='path to filter data of queries')
    parser.add_argument('--result-path', type=str, default="./gt.tsv",
                        help='path to save ground truth')


    args = parser.parse_args()

    gen(args.passage_path_prefix, args.passage_filter_path, args.query_path, args.query_filter_path, args.result_path)
