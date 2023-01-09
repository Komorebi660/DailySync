import struct
import pickle
import numpy as np
import argparse


def pack_corpus(corpus_path, corpus_bin_path, corpus_base_bin_path):
    embeddings = []
    fvec = open(corpus_bin_path, "wb")
    fvec.write(struct.pack("i", 8841823))   # number of documents
    fvec.write(struct.pack("i", 769))       # dimension of vectors
    for i in range(10):
        with open(corpus_path + "%d.pt" % i, "rb") as f:
            embedding, _ = pickle.load(f)
            embeddings.append(embedding)

            length = embedding.shape[0]
            embedding_flattern = embedding.flatten()
            fvec.write(struct.pack("%sf" % (length*769), *embedding_flattern))
            print("loaded split %d, length = %d" % (i, length))
    fvec.close()
    print("finish corpus write")
    embeddings = np.vstack(embeddings)
    print(embeddings.shape)

    fvec = open(corpus_base_bin_path, "wb")
    fvec.write(struct.pack("i", 841823))    # number of documents
    fvec.write(struct.pack("i", 769))       # dimension of vectors
    embedding_flattern = embeddings[:841823].flatten()
    fvec.write(struct.pack("%sf" % (841823*769), *embedding_flattern))
    fvec.close()
    print("finish corpus base write")


def pack_query(query_path, query_bin_path):
    fvec = open(query_bin_path, "wb")
    fvec.write(struct.pack("i", 6980))      # number of documents
    fvec.write(struct.pack("i", 769))       # dimension of vectors
    with open(query_path, "rb") as f:
        embedding, _ = pickle.load(f)
        #flattern embedding
        embedding_flattern = embedding.flatten()
        #write in bytes
        fvec.write(struct.pack("%sf" % (6980*769), *embedding_flattern))
    fvec.close()


# use to verify
def unpack(path):
    with open(path, "rb") as f:
        num_doc = struct.unpack("i", f.read(4))[0]
        dim = struct.unpack("i", f.read(4))[0]
        print(num_doc, dim)
        queries = np.frombuffer(f.read(num_doc * dim * 4), dtype=np.float32).reshape((num_doc, dim))
        print(queries)
        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--passage-path-prefix', type=str, default="../embedding_data/corpus/split0",
                        help='prefix of path to passage embeddings')
    parser.add_argument('--query-path', type=str, default="../embedding_data/query/query_dev_small.pt",
                        help='path to query embeddings')
    parser.add_argument('--corpus-bin-path', type=str, default="doc_vectors.bin",
                        help='path to bin data of passages(8.8M)')
    parser.add_argument('--corpus-base-bin-path', type=str, default="doc_vectors_base.bin",
                        help='path to bin data of base passages(800K)')
    parser.add_argument('--query-bin-path', type=str, default="query_vectors.bin",
                        help='path to to bin data of queries')
    args = parser.parse_args()

    pack_query(args.query_path, args.query_bin_path)
    pack_corpus(args.passage_path_prefix, args.corpus_bin_path, args.corpus_base_bin_path)
