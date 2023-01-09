import struct
import csv
import argparse


def transform(query_path, bin_path, tsv_path):
    with open(query_path, "r", encoding="utf8") as f_query, \
            open(bin_path, "rb") as f:
        tsvreader = csv.reader(f_query, delimiter="\t")

        num_query = struct.unpack("i", f.read(4))[0]
        num_result = struct.unpack("i", f.read(4))[0]
        print(num_query, num_result)

        with open(tsv_path, "w", encoding="utf8") as f_result:
            for [qid, _] in tsvreader:
                for rank in range(num_result):
                    pid = struct.unpack("i", f.read(4))[0]
                    score = struct.unpack("f", f.read(4))[0]
                    f_result.write(f"{qid} 0 {pid} {rank+1} {100.0-score} IndriQueryLikelihood\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--query-path', type=str, default="/data/data5/v-yaoqichen/data/queries.dev.small.tsv",
                        help='path to filter data of passages')
    parser.add_argument('--bin-result-path', type=str, default="./results/result-8k",
                        help='path to binary search result which need to be transfered')
    parser.add_argument('--tsv-result-path', type=str, default="./results/result-8k.tsv",
                        help='path to tsv search result')

    args = parser.parse_args()

    transform(args.query_path, args.bin_result_path, args.tsv_result_path)
