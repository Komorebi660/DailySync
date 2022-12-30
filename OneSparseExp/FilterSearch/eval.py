import csv
import numpy as np


def eval(gt_path, result_path):
    recall_1, recall_10, recall_100 = 0.0, 0.0, 0.0
    mrr_10, mrr_100 = 0.0, 0.0
    i = 0
    with open(gt_path, "r", encoding="utf8") as f_gt, \
            open(result_path, "r", encoding="utf8") as f_result:
        gt_tsvreader = csv.reader(f_gt, delimiter=" ")
        result_tsvreader = csv.reader(f_result, delimiter="\t")
        gt_list = []
        result_list = []
        result_list_with_rank = []
        for [_, gt_docid, _, _], [_, docid, rank, _] in zip(gt_tsvreader, result_tsvreader):
            i += 1
            gt_list.append(gt_docid)
            result_list.append(docid)
            result_list_with_rank.append([docid, rank])
            if(i % 100 == 0):
                #calculate recall
                recall_1 += len(set(gt_list).intersection(set(result_list[:1])))/1
                recall_10 += len(set(gt_list).intersection(set(result_list[:10])))/10
                recall_100 += len(set(gt_list).intersection(set(result_list[:100])))/100
                #calculate mrr
                for [docid, rank] in result_list_with_rank:
                    if docid in gt_list:
                        mrr_100 += 1.0/int(rank)
                        break
                for [docid, rank] in result_list_with_rank[:10]:
                    if docid in gt_list:
                        mrr_10 += 1.0/int(rank)
                        break
                #clear
                gt_list = []
                result_list = []
                result_list_with_rank = []

    print(f"recall@1: {recall_1*100/i}")
    print(f"recall@10: {recall_10*100/i}")
    print(f"recall@100: {recall_100*100/i}")
    print(f"mrr@10: {mrr_10*100/i}")
    print(f"mrr@100: {mrr_100*100/i}")


if __name__ == "__main__":
    eval("./gt.tsv", "./msmarco_spann_filter_qrels.tsv")
