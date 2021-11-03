'''
python merge_tagged_datasets.py \
    --tagged-positives-file tagged_dataset_positives.jsonl \
    --tagged-negatives-file tagged_dataset_negatives.jsonl \
    --negative-mining sample \
    --num-negatives 7
'''

import argparse
import jsonlines
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('--tagged-positives-file', type=str, default="tagged_dataset_positives.jsonl")
parser.add_argument('--tagged-negatives-file', type=str, default="tagged_dataset_negatives.jsonl")
parser.add_argument('--combined-file', type=str, default="tagged_datasets_merged.jsonl")
parser.add_argument('--dataset-tldrs', type=str, default="tagged_dataset_tldrs.hypo")
parser.add_argument('--negative-mining', choices=["sample", "hard"], default="sample")
parser.add_argument('--num-negatives', type=int, default=7, help="How many negatives to choose for each dataset")

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


if __name__ == "__main__":
    args = parser.parse_args()
    tagged_positives = load_rows(args.tagged_positives_file)
    tagged_negatives = load_rows_by_abstract(args.tagged_negatives_file)
    tldrs = open(args.dataset_tldrs).read().split("\n")[:-1]

    for i in range(len(tagged_positives)):
        all_negatives = tagged_negatives[tagged_positives[i]["abstract"]]["datasets"]
        tagged_positives[i]["positives"] = tagged_positives[i]["datasets"]
        del tagged_positives[i]["datasets"]
        if len(all_negatives) > args.num_negatives:
            if args.negative_mining == "sample":
                tagged_positives[i]["negatives"] = list(np.random.choice(all_negatives, size=args.num_negatives, replace=False))
            else:
                raise NotImplementedError
        tagged_positives[i]["tldr"] = tldrs[i]
    write_rows(tagged_positives, args.combined_file)