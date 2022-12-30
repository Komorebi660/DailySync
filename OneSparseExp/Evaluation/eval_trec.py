# https://pypi.org/project/pytrec-eval/

import argparse
import logging
import json
import sys
sys.path.insert(0, '')

import pytrec_eval
import re
from typing import List, Dict, Union, Tuple
ROUND_DIGIT=6

def calc_ndcg(qrels: Dict[str, Dict[str, int]],
        results: Dict[str, Dict[str, float]],
        k_values: List[int]) -> Tuple[Dict[str, float]]:
    ndcg = {}
    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {ndcg_string, })
    scores = evaluator.evaluate(results)
    for query_id in scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / len(scores), ROUND_DIGIT)
    return ndcg


def calc_recall_official(qrels: Dict[str, Dict[str, int]],
        results: Dict[str, Dict[str, float]],
        k_values: List[int]) -> Tuple[Dict[str, float]]:
    recall = {}
    for k in k_values:
        recall[f"recall_official@{k}"] = 0.0
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {recall_string, })
    scores = evaluator.evaluate(results)
    for query_id in scores.keys():
        for k in k_values:
            recall[f"recall_official@{k}"] += scores[query_id]["recall_" + str(k)]
    for k in k_values:
        recall[f"recall_official@{k}"] = round(recall[f"recall_official@{k}"] / len(scores), ROUND_DIGIT)
    return recall

def calc_mrr_official(qrels: Dict[str, Dict[str, int]],
        results: Dict[str, Dict[str, float]],
        k_values: List[int]) -> Tuple[Dict[str, float]]:
    MRR = dict()

    top_hits = dict()
    k_max = max(k_values)
    for query_id, doc_scores in results.items():
        top_hits[query_id] = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"recip_rank", })
    for k in k_values:
        local_results = dict((query_id, dict((k, v) for k, v in ths[:k])) for query_id, ths in top_hits.items())
        scores = evaluator.evaluate(local_results)
        all_scores = [scores[query_id]["recip_rank"] for query_id in scores]
        MRR[f"MRR_official@{k}"] = round(sum(all_scores) / len(all_scores), ROUND_DIGIT)
    return MRR


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--trec_path", type=str, required=True)
    parser.add_argument("--qrels_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    # 1. load TREC-formatted result file
    logging.info(f"loading trec {args.trec_path}")
    qid2results = dict()  # {qid: {did: score}}
    with open(args.trec_path) as fp:
        for idx_line, line in enumerate(fp):
            try:
                qid, _, did, rank, score, _method = re.split('\t| ', line.strip("\n"))
            except ValueError:
                print(f"Please check line {idx_line} in {args.trec_path}")
                raise ValueError
            qid, did, score = qid, did, float(score)
            if qid not in qid2results:  # initialize
                qid2results[qid] = dict()
            qid2results[qid][did] = score

    # 2. load QRELs file
    qrels = dict()
    with open(args.qrels_path) as fp:
        for line in fp:
            if "\t" in line:
                data = line.strip().split("\t")
            else:
                data = line.strip().split(" ")
            try:
                qsid, psid, gold_score = str(data[0]), str(data[2]), int(data[3])
            except IndexError:
                print(line)
                print(data)
                exit(0)
            if qsid not in qrels:
                qrels[qsid] = dict()
            qrels[qsid][psid] = gold_score

    # 3. sanity check
    if len(qrels) != len(qid2results):
        print(f"Warning: qrels (len is {len(qrels)}) is not equal to results (len is {len(qid2results)})")
    results = dict((k, v) for k, v in qid2results.items() if k in qrels)

    # 4.calculate ndcg, mrr, recall
    res_metrics = {"QueriesRanked": len(results)}
    res_metrics.update(calc_mrr_official(qrels, results, [10, 100]))
    res_metrics.update(calc_recall_official(qrels, results, [1, 10, 100]))
    res_metrics.update(calc_ndcg(qrels, results, [10, 100]))

    logging.info(res_metrics)
    if args.output_path is not None:
        with open(args.output_path, "w") as fp:
            json.dump(res_metrics, fp)

