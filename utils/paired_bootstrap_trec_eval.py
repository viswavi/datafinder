'''
python bootstrap_trec_eval.py \
    /projects/ogma1/vijayv/dataset-recommendation/test_dataset_collection.qrels \
    /projects/ogma1/vijayv/dataset-recommendation/bm25_keyphrase_with_structured_information_4_limit.trec
'''

import argparse
from collections import defaultdict
import numpy as np
import os
import subprocess
import tempfile
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('relevance_judgements_file_a', type=str, help='Query relevance judgements file')
parser.add_argument('trec_file_a', type=str, help='Output in TREC format')
parser.add_argument('relevance_judgements_file_b', type=str, help='Query relevance judgements file')
parser.add_argument('trec_file_b', type=str, help='Output in TREC format')
parser.add_argument('--num-samples', type=int, default=1000)

def construct_subsampled_results(gt_lines_by_query_a, ret_lines_by_query_a, gt_lines_by_query_b, ret_lines_by_query_b, query_ids, num_samples=1000, sample_ratio=0.5):
    subsamples = []
    for _ in range(num_samples):
        sampled_qids = np.random.choice(query_ids, int(len(query_ids)*sample_ratio), replace=True)
        gt_lines_subset_a = {}
        ret_lines_subset_a = {}
        gt_lines_subset_b = {}
        ret_lines_subset_b = {}
        for qid in sampled_qids:
            if qid in gt_lines_by_query_a:
                gt_lines_subset_a[qid] = (gt_lines_by_query_a[qid])
            if qid in gt_lines_by_query_b:
                gt_lines_subset_b[qid] = (gt_lines_by_query_b[qid])
            if qid in ret_lines_by_query_a:
                ret_lines_subset_a[qid] = (ret_lines_by_query_a[qid])
            if qid in ret_lines_by_query_b:
                ret_lines_subset_b[qid] = (ret_lines_by_query_b[qid])
        subsamples.append((gt_lines_subset_a, ret_lines_subset_a, gt_lines_subset_b, ret_lines_subset_b))
    return subsamples

def write_file_from_dict(gt_query_dict, header_row, fname):
    outfile = open(fname, 'w')
    outlines = [header_row]
    for qid in sorted(list(gt_query_dict.keys())):
        rows = gt_query_dict[qid]
        outlines.extend(rows)
    outfile.write("\n".join(outlines))
    outfile.close()


def accumulate_metrics_dicts(metrics_dicts):
    accumulated_dicts = defaultdict(list)
    for md in metrics_dicts:
        for k, v in md.items():
            accumulated_dicts[k].append(v)
    return accumulated_dicts

def compute_p_betters(accumulated_metrics_dict_a, accumulated_metrics_dict_b):
    p_better_dict = {}
    assert accumulated_metrics_dict_a.keys() == accumulated_metrics_dict_b.keys()
    for k, vals_a in accumulated_metrics_dict_a.items():
        vals_b = accumulated_metrics_dict_b[k]
        num_better = 0
        num_worse = 0
        for a, b in zip(vals_a, vals_b):
            if a > b:
                num_better += 1
            else:
                num_worse += 1
        p_better_dict[k] = (num_better, num_worse)
    return p_better_dict

if __name__ == "__main__":
    args = parser.parse_args()
    tempdir = tempfile.TemporaryDirectory()

    gt_lines_by_query_a = defaultdict(list)
    ret_lines_by_query_a = defaultdict(list)
    gt_lines_by_query_b = defaultdict(list)
    ret_lines_by_query_b = defaultdict(list)

    relevance_judgements_lines = [t.strip() for t in open(args.relevance_judgements_file_a).readlines()]
    relevance_header = relevance_judgements_lines[0]
    for line in relevance_judgements_lines[1:]:
        query_id = line.split("\t")[0]
        gt_lines_by_query_a[query_id].append(line)

    trec_lines = [t.strip() for t in open(args.trec_file_a).readlines()]
    trec_header = trec_lines[0]
    for line in trec_lines[1:]:
        query_id = line.split("\t")[0]
        ret_lines_by_query_a[query_id].append(line)

    relevance_judgements_lines = [t.strip() for t in open(args.relevance_judgements_file_b).readlines()]
    for line in relevance_judgements_lines[1:]:
        query_id = line.split("\t")[0]
        gt_lines_by_query_b[query_id].append(line)

    trec_lines = [t.strip() for t in open(args.trec_file_b).readlines()]
    for line in trec_lines[1:]:
        query_id = line.split("\t")[0]
        ret_lines_by_query_b[query_id].append(line)

    all_query_ids = list(set(gt_lines_by_query_a.keys()).union(ret_lines_by_query_a.keys()).union(gt_lines_by_query_b.keys()).union(ret_lines_by_query_b.keys()))
    query_id_bootstrap_samples = construct_subsampled_results(gt_lines_by_query_a, ret_lines_by_query_a, gt_lines_by_query_b, ret_lines_by_query_b, all_query_ids, num_samples=args.num_samples)

    metrics_dicts_a = []
    metrics_dicts_b = []
    for i, (gt_dict_a, ret_dict_a, gt_dict_b, ret_dict_b) in tqdm(enumerate(query_id_bootstrap_samples)):
        gt_file_a = os.path.join(tempdir.name, str(i) + "_a.qrels")
        ret_file_a = os.path.join(tempdir.name, str(i) + "_a.trec")
        gt_file_b = os.path.join(tempdir.name, str(i) + "_b.qrels")
        ret_file_b = os.path.join(tempdir.name, str(i) + "_b.trec")
        write_file_from_dict(gt_dict_a, relevance_header, gt_file_a)
        write_file_from_dict(ret_dict_a, trec_header, ret_file_a)
        write_file_from_dict(gt_dict_b, relevance_header, gt_file_b)
        write_file_from_dict(ret_dict_b, trec_header, ret_file_b)
        commands_a = ["./trec_eval", "-c", "-m", "P.5", "-m", "recall.5", "-m", "map", "-m", "recip_rank", gt_file_a, ret_file_a]
        commands_b = ["./trec_eval", "-c", "-m", "P.5", "-m", "recall.5", "-m", "map", "-m", "recip_rank", gt_file_b, ret_file_b]
        p = subprocess.Popen(commands_a, stdout=subprocess.PIPE, cwd="/projects/ogma1/vijayv/anserini/tools/eval/trec_eval.9.0.4/")
        output_a = p.stdout.read().decode(encoding='utf-8')
        metrics_dict_a = {}
        for row in output_a.split("\n"):
            if len(row.strip()) == 0:
                continue
            cols = row.strip().split("\t")
            metrics_dict_a[cols[0].strip()] = float(cols[-1])
        metrics_dicts_a.append(metrics_dict_a)

        p = subprocess.Popen(commands_b, stdout=subprocess.PIPE, cwd="/projects/ogma1/vijayv/anserini/tools/eval/trec_eval.9.0.4/")
        output_b = p.stdout.read().decode(encoding='utf-8')
        metrics_dict_b = {}
        for row in output_b.split("\n"):
            if len(row.strip()) == 0:
                continue
            cols = row.strip().split("\t")
            metrics_dict_b[cols[0].strip()] = float(cols[-1])
        metrics_dicts_b.append(metrics_dict_b)

    accumulated_dicts_a = accumulate_metrics_dicts(metrics_dicts_a)
    accumulated_dicts_b = accumulate_metrics_dicts(metrics_dicts_b)

    def sorting_order(k):
        if k == "P_5":
            return 1
        elif k == "recall_5":
            return 2
        elif k == "map":
            return 3
        elif k == "recip_rank":
            return 4
        else:
            raise ValueError(f"Unexpected metric key {k}")

    p_better_dict = compute_p_betters(accumulated_dicts_a, accumulated_dicts_b)

    print(f"System A\nmetric\t\t\tmean\t\t\tstddev")
    for k in sorted(list(accumulated_dicts_a.keys()), key=lambda x: sorting_order(x)):
        mean = np.mean(accumulated_dicts_a[k])
        std = np.std(accumulated_dicts_a[k])
        if k == "recall_5":
            tabs = "\t\t"
        else:
            tabs = "\t\t\t"
        p_better = p_better_dict[k][0]
        print(f"{k}{tabs}{round(mean, 6)}\t\t\t{round(std, 6)}\t\t\t\t\t\tP(better): {round(p_better, 3)}")
    
    print(f"System B\nmetric\t\t\tmean\t\t\tstddev")
    for k in sorted(list(accumulated_dicts_b.keys()), key=lambda x: sorting_order(x)):
        mean = np.mean(accumulated_dicts_b[k])
        std = np.std(accumulated_dicts_b[k])
        if k == "recall_5":
            tabs = "\t\t"
        else:
            tabs = "\t\t\t"
        p_better = p_better_dict[k][1]
        print(f"{k}{tabs}{round(mean, 6)}\t\t\t{round(std, 6)}\t\t\tP(better): {round(p_better, 3)}")

