import csv
import argparse


def eval(gt_path, result_path, k):
    recall = 0.0
    mrr = 0.0
    no_match = 0
    with open(gt_path, "r", encoding="utf8") as f_gt, \
         open(result_path, "r", encoding="utf8") as f_result:
        gt_tsvreader = csv.reader(f_gt, delimiter="\t")
        result_tsvreader = csv.reader(f_result, delimiter="\t")
        
        gt_dict = {}
        #for [gt_qid, gt_docid, _, _] in gt_tsvreader:
        for [gt_qid, gt_docid] in gt_tsvreader:
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

        idx = 0
        for qid in gt_dict.keys():
            gt_list = gt_dict[qid]
            idx += 1
            if qid not in result_dict:
                no_match += 1
                continue
            result_list = result_dict[qid]
            # calculate recall
            _recall = len(set(gt_list).intersection(set(result_list)))/len(set(gt_list))
            recall += _recall
            if(_recall < 1.0):
                no_match += 1
            # calculate mrr
            for i in range(len(result_list)):
                if result_list[i] in gt_list:
                    mrr += 1.0/(i+1)
                    break

    print(f"recall@{k}: {recall/idx}")
    print(f"mrr@{k}: {mrr/idx}")
    print(f"no match number:{no_match}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt-path', type=str, default="./gt.tsv",
                        help='path to filter data of passages')
    parser.add_argument('--search-result-path', type=str, default="./qrels.tsv",
                        help='path to save search result')
    parser.add_argument('--k', type=int, default=100,
                        help='evaluate top k results')

    args = parser.parse_args()

    eval(args.gt_path, args.search_result_path, args.k)
