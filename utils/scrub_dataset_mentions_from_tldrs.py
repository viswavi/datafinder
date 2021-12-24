'''
python scrub_dataset_mentions_from_tldrs.py train_tldrs.hypo train_tldrs_scrubbed.hypo
'''

import argparse
import json
import jsonlines
import sys
sys.path.extend(["..", "."])
from tqdm import tqdm
from utils.utils import scrub_dataset_references



parser = argparse.ArgumentParser()
parser.add_argument('in_file', type=str, help='Input TLDRs to process')
parser.add_argument('out_file', type=str, help='Processed TLDRs path')
parser.add_argument('--positive-dataset-tags', type=str, default="tagged_dataset_positives.jsonl", help="List of datasets tagged per training paper")
parser.add_argument('--datasets-file', type=str, default="datasets.json", help="JSON file containing metadata about all datasets on PapersWithCode")

if __name__ == "__main__":
    args = parser.parse_args()
    out_rows = []
    tldrs = [t.strip() for t in open(args.in_file).readlines()]
    positive_dataset_tags = list(jsonlines.open(args.positive_dataset_tags))
    assert len(positive_dataset_tags) == len(tldrs), breakpoint()
    datasets_list = json.load(open(args.datasets_file))
    for i, tldr in tqdm(enumerate(tldrs)):
        dataset_names = []
        for dataset in datasets_list:
            if dataset["name"] in positive_dataset_tags[i]["datasets"]:
                dataset_names.append(dataset["name"])
                # dataset_names.extend(dataset["variants"])
        dataset_names = list(set(dataset_names))

        if len(tldr.split()) == 0:
            out_rows.append(tldr)
        else:
            scrubbed = scrub_dataset_references(tldr, dataset_names)
            out_rows.append(scrubbed)
    outfile = open(args.out_file, 'w')
    outfile.write("\n".join(out_rows))
    outfile.close()