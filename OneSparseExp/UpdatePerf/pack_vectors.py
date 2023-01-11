import struct
import pickle
import numpy as np
import argparse

suffix = ["841K", "1.77M", "2.65M", "3.54M", "4.42M", "5.31M", "6.19M", "7.07M", "7.96M", "8.84M"]
length = [884183, 1768366, 2652549, 3536732, 4420915, 5305098, 6189281, 7073464, 7957647, 8841823]


def pack_corpus(passage_path_prefix, corpus_bin_path_prefix, start=0, end=10):
    for i in range(start, end):
        print("start paking corpus %s" % suffix[i])
        embeddings = []
        fvec = open(corpus_bin_path_prefix + "%s.bin" % suffix[i], "wb")
        fvec.write(struct.pack("i", length[i]))     # number of documents
        fvec.write(struct.pack("i", 769))           # dimension of vectors
        total_size = 0
        for j in range(i+1):
            with open(passage_path_prefix + "%d.pt" % j, "rb") as f:
                embedding, _ = pickle.load(f)
                embeddings.append(embedding)

                size = embedding.shape[0]
                total_size += size
                embedding_flattern = embedding.flatten()
                fvec.write(struct.pack("%sf" % (size*769), *embedding_flattern))
            print("pack %d" % j)
        fvec.close()
        print("finish corpus %d, size = %d" % (i, total_size))


def pack_query(query_path, query_bin_path):
    fvec = open(query_bin_path, "wb")
    fvec.write(struct.pack("i", 6980))      # number of documents
    fvec.write(struct.pack("i", 769))       # dimension of vectors
    with open(query_path, "rb") as f:
        embedding, _ = pickle.load(f)
        # flattern embedding
        embedding_flattern = embedding.flatten()
        #write in bytes
        fvec.write(struct.pack("%sf" % (6980*769), *embedding_flattern))
    fvec.close()


def unpack(path):
    with open(path, "rb") as f:
        num_doc = struct.unpack("i", f.read(4))[0]
        dim = struct.unpack("i", f.read(4))[0]
        print(num_doc, dim)
        embeddings = np.frombuffer(f.read(num_doc * dim * 4), dtype=np.float32).reshape((num_doc, dim))
        print(embeddings.shape)
        # print(embeddings)
        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--passage-path-prefix', type=str, default="../embedding_data/corpus/split0",
                        help='prefix of path to passage embeddings')
    parser.add_argument('--query-path', type=str, default="../embedding_data/query/query_dev_small.pt",
                        help='path to query embeddings')
    parser.add_argument('--corpus-bin-path-prefix', type=str, default="./doc_vectors_",
                        help='prefix of path to bin data of passages')
    parser.add_argument('--query-bin-path', type=str, default="./query_vectors.bin",
                        help='path to to bin data of queries')
    args = parser.parse_args()

    pack_query(args.query_path, args.query_bin_path)
    unpack(args.query_bin_path)
    pack_corpus(args.passage_path_prefix, args.corpus_bin_path_prefix)
    for i in range(10):
        unpack(args.corpus_bin_path_prefix + "%s.bin" % suffix[i])
