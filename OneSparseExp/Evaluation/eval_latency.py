import csv
import argparse
import numpy as np


def calculate_latency(path):
    with open(path, "r", encoding="utf8") as f:
        tsvreader = csv.reader(f, delimiter="\t")
        idx = 0
        temp = 0.0
        latency_list = []
        for [qid, latency] in tsvreader:
            idx += 1
            temp += float(latency)
            latency_list.append(float(latency))

        latency_numpy_list = np.array(latency_list)
        print(np.percentile(latency_numpy_list, [50, 90, 99]))
        print("Latency: ", temp / idx)
        print("QPS: ", idx / temp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./latency.tsv',
                        help='path to latency result')
    args = parser.parse_args()
    calculate_latency(args.path)
