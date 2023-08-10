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
parser.add_argument('relevance_judgements_file', type=str, help='Query relevance judgements file')
parser.add_argument('trec_file', type=str, help='Output in TREC format')
parser.add_argument('--num-samples', type=int, default=1000)

def construct_subsampled_results(gt_lines_by_query, ret_lines_by_query, query_ids, num_samples=1000, sample_ratio=0.5):
    subsamples = []
    for _ in range(num_samples):
        sampled_qids = np.random.choice(query_ids, int(len(query_ids)*sample_ratio), replace=True)
        gt_lines_subset = {}
        ret_lines_subset = {}
        for qid in sampled_qids:
            if qid in gt_lines_by_query:
                gt_lines_subset[qid] = (gt_lines_by_query[qid])
            if qid in ret_lines_by_query:
                ret_lines_subset[qid] = (ret_lines_by_query[qid])
        subsamples.append((gt_lines_subset, ret_lines_subset))
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

if __name__ == "__main__":
    args = parser.parse_args()
    tempdir = tempfile.TemporaryDirectory()

    gt_lines_by_query = defaultdict(list)
    ret_lines_by_query = defaultdict(list)

    relevance_judgements_lines = [t.strip() for t in open(args.relevance_judgements_file).readlines()]
    relevance_header = relevance_judgements_lines[0]
    for line in relevance_judgements_lines[1:]:
        query_id = line.split("\t")[0]
        gt_lines_by_query[query_id].append(line)

    trec_lines = [t.strip() for t in open(args.trec_file).readlines()]
    trec_header = trec_lines[0]
    for line in trec_lines[1:]:
        query_id = line.split("\t")[0]
        ret_lines_by_query[query_id].append(line)

    all_query_ids = list(set(gt_lines_by_query.keys()).union(ret_lines_by_query.keys()))
    query_id_bootstrap_samples = construct_subsampled_results(gt_lines_by_query, ret_lines_by_query, all_query_ids, num_samples=args.num_samples)

    metrics_dicts = []
    for i, (gt_dict, ret_dict) in tqdm(enumerate(query_id_bootstrap_samples)):
        gt_file = os.path.join(tempdir.name, str(i) + ".qrels")
        ret_file = os.path.join(tempdir.name, str(i) + ".trec")
        write_file_from_dict(gt_dict, relevance_header, gt_file)
        write_file_from_dict(ret_dict, trec_header, ret_file)
        commands = ["./trec_eval", "-c", "-m", "P.5", "-m", "recall.5", "-m", "map", "-m", "recip_rank", gt_file, ret_file]
        string=[]
        p = subprocess.Popen(commands, stdout=subprocess.PIPE, cwd="/projects/ogma1/vijayv/anserini/tools/eval/trec_eval.9.0.4/")
        output = p.stdout.read().decode(encoding='utf-8')
        metrics_dict = {}
        for row in output.split("\n"):
            if len(row.strip()) == 0:
                continue
            cols = row.strip().split("\t")
            metrics_dict[cols[0].strip()] = float(cols[-1])
        metrics_dicts.append(metrics_dict)

    accumulated_dicts = accumulate_metrics_dicts(metrics_dicts)

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
    
    print(f"metric\t\t\tmean\t\t\tstddev")
    for k in sorted(list(accumulated_dicts.keys()), key=lambda x: sorting_order(x)):
        mean = np.mean(accumulated_dicts[k])
        std = np.std(accumulated_dicts[k])
        print(f"{k}\t\t\t{round(mean, 3)}\t\t\t{round(std, 3)}")

