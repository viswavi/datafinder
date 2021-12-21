'''
python merge_tagged_datasets.py \
    --combined-file tagged_datasets_merged_random_negatives.jsonl \
    --dataset-tldrs tagged_dataset_tldrs_scrubbed.hypo \
    --tagged-positives-file tagged_dataset_positives.jsonl \
    --tagged-negatives-file tagged_dataset_negatives.jsonl \
    --negative-mining sample \
    --num-negatives 7

or


python merge_tagged_datasets.py \
    --combined-file tagged_datasets_merged_hard_negatives.jsonl \
    --dataset-tldrs train_tldrs_scrubbed.hypo \
    --tagged-positives-file tagged_dataset_positives.jsonl \
    --tagged-negatives-file tagged_dataset_negatives.jsonl \
    --negative-mining hard \
    --anserini-index indexes/dataset_search_collection_no_paper_text_jsonl \
    --num-negatives 7
'''

import argparse
import jsonlines
import numpy as np
import os
from pyserini.search import SimpleSearcher

parser = argparse.ArgumentParser()
parser.add_argument('--tagged-positives-file', type=str, default="tagged_dataset_positives.jsonl")
parser.add_argument('--tagged-negatives-file', type=str, default="tagged_dataset_negatives.jsonl")
parser.add_argument('--combined-file', type=str, default="tagged_datasets_merged.jsonl")
parser.add_argument('--dataset-tldrs', type=str, default="train_tldrs_scrubbed.hypo")
parser.add_argument('--num-negatives', type=int, default=7, help="How many negatives to choose for each dataset")
parser.add_argument('--negative-mining', choices=["sample", "hard"], default="sample")
parser.add_argument('--anserini-index', type=str, help="Path to Anserini index of datasets to search to find hard negatives")

def load_rows(search_collection_file):
    return list(jsonlines.open(search_collection_file))

def write_rows(rows, outfile):
    with open(outfile, 'w') as f:
        writer = jsonlines.Writer(f)
        writer.write_all(rows)
    print(f"Wrote {len(rows)} rows to {outfile}.")

def load_rows_by_abstract(search_collection):
    rows = {} 
    for doc in jsonlines.open(search_collection):
        rows[doc["abstract"]] = doc
    return rows

def find_hard_negatives(searcher, query, results_limit, all_negatives):
    '''
    Return top `results_limit` datasets that aren't a true positive
    '''
    hits = searcher.search(query)
    hardest_negatives = []
    rank = 0
    for hit in hits:
        if hit.docid in hardest_negatives or hit.docid not in all_negatives:
            print(f"Negative {hit.docid} ignored because it is not contained in all_negatives")
            continue
        else:
            hardest_negatives.append(hit.docid)
            rank += 1
        if rank == results_limit:
            break
    return hardest_negatives

if __name__ == "__main__":
    args = parser.parse_args()
    tagged_positives = load_rows(args.tagged_positives_file)
    tagged_negatives = load_rows_by_abstract(args.tagged_negatives_file)
    tldrs = [t.strip() for t in open(args.dataset_tldrs).readlines()]

    if args.negative_mining == "hard":
        # Mine negatives via BM25 retrieval
        searcher = SimpleSearcher(args.anserini_index)
        # ^ Use default BM25 parameters for now
        # searcher.set_bm25(k1=0.8, b=0.4)

    for i in range(len(tagged_positives)):
        instance_abstract = tagged_positives[i]["abstract"]
        all_negatives = tagged_negatives[instance_abstract]["datasets"]
        tagged_positives[i]["tldr"] = tldrs[i]
        tagged_positives[i]["positives"] = tagged_positives[i]["datasets"]
        del tagged_positives[i]["datasets"]
        if len(all_negatives) > args.num_negatives:
            if args.negative_mining == "sample":
                tagged_positives[i]["negatives"] = list(np.random.choice(all_negatives, size=args.num_negatives, replace=False))
            elif args.negative_mining == "hard":
                tagged_positives[i]["negatives"] = find_hard_negatives(searcher, tldrs[i], args.num_negatives, all_negatives)
            else:
                raise NotImplementedError
    write_rows(tagged_positives, args.combined_file)