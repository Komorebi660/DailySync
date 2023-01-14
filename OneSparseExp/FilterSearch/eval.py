import csv
import argparse


def eval(gt_path, result_path, k):
    recall= 0.0
    mrr = 0.0
    with open(gt_path, "r", encoding="utf8") as f_gt, \
            open(result_path, "r", encoding="utf8") as f_result:
        gt_tsvreader = csv.reader(f_gt, delimiter="\t")
        result_tsvreader = csv.reader(f_result, delimiter="\t")
        gt_dict = {}
        for [gt_qid, gt_docid, _, _] in gt_tsvreader:
            if int(gt_qid) not in gt_dict.keys():
                gt_dict[int(gt_qid)] = [gt_docid]
            else:
                gt_dict[int(gt_qid)].append(gt_docid)
        result_dict = {}
        for [result_qid, result_docid, _, _] in result_tsvreader:
            if int(result_qid) not in result_dict.keys():
                result_dict[int(result_qid)] = [result_docid]
            else:
                result_dict[int(result_qid)].append(result_docid)

        for qid in gt_dict.keys():
            gt_list = gt_dict[qid]
            result_list = result_dict[qid]
            #calculate recall
            recall += len(set(gt_list).intersection(set(result_list[:k])))/k
            #calculate mrr
            _k = k if k < len(result_list) else len(result_list)
            for i in range(_k):
                if result_list[i] in gt_list:
                    mrr += 1.0/(i+1)
                    break

    print(f"recall@{k}: {recall/6980}")
    print(f"mrr@{k}: {mrr/6980}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt-path', type=str, default="./gt.tsv",
                        help='path to filter data of passages')
    parser.add_argument('--search-result-path', type=str, default="./spann_filter_qrels.tsv",
                        help='path to save search result')
    parser.add_argument('--k', type=int, default=100,
                        help='evaluate top k results')

    args = parser.parse_args()

    eval(args.gt_path, args.search_result_path, args.k)
