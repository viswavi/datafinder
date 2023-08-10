'''
python data_processing/train_data/merge_tagged_datasets.py \
    --combined-file tagged_datasets_merged_random_negatives.jsonl \
    --dataset-tldrs tagged_dataset_tldrs_scrubbed.hypo \
    --tagged-positives-file tagged_dataset_positives.jsonl \
    --tagged-negatives-file tagged_dataset_negatives.jsonl \
    --negative-mining sample \
    --num-negatives 7

or

python data_processing/train_data/merge_tagged_datasets.py \
    --combined-file tagged_datasets_merged_hard_negatives.jsonl \
    --dataset-tldrs train_tldrs_scrubbed.hypo \
    --tagged-positives-file tagged_dataset_positives.jsonl \
    --tagged-negatives-file tagged_dataset_negatives.jsonl \
    --negative-mining hard \
    --anserini-index indexes/dataset_search_collection_no_paper_text_jsonl \
    --num-negatives 7
'''

import argparse
import json
import jsonlines
import numpy as np
import os
import nltk
from pyserini.search import SimpleSearcher
import sys
sys.path.extend(["..", ".", "../.."])
from tqdm import tqdm
from utils.utils import scrub_dataset_references

stopwords = nltk.corpus.stopwords.words("english")

parser = argparse.ArgumentParser()
parser.add_argument('--tagged-positives-file', type=str, default="tagged_dataset_positives.jsonl")
parser.add_argument('--tagged-negatives-file', type=str, default="tagged_dataset_negatives.jsonl")
parser.add_argument('--datasets-file', type=str, default="data/datasets_01_15_2023.json")
parser.add_argument('--combined-file', type=str, default="tagged_datasets_merged.jsonl")
parser.add_argument('--parse-file', type=str, default="data/train_abstracts_parsed_by_galactica.jsonl")
parser.add_argument('--num-negatives', type=int, default=7, help="How many negatives to choose for each dataset")
parser.add_argument('--negative-mining', choices=["sample", "hard"], default="sample")
parser.add_argument('--anserini-index', type=str, help="Path to Anserini index of datasets to search to find hard negatives")
parser.add_argument('--keyphrases-to-use', type=str, default="task,modality,domain,language,length")


def load_rows(search_collection_file):
    return list(jsonlines.open(search_collection_file))

def write_rows(rows, outfile):
    with open(outfile, 'w') as f:
        writer = jsonlines.Writer(f)
        writer.write_all(rows)
    print(f"Wrote {len(rows)} rows to {outfile}.")

def load_rows_by_abstract(search_collection):
    rows = {} 
    for doc in tqdm(jsonlines.open(search_collection)):
        rows[doc["abstract"]] = doc
    return rows

def find_hard_negatives(searcher, query, results_limit, all_negatives):
    '''
    Return top `results_limit` datasets that aren't a true positive
    '''
    hits = searcher.search(query, k=500)
    hardest_negatives = []
    rank = 0
    for hit in hits:
        if hit.docid in hardest_negatives or hit.docid not in all_negatives:
            continue
        else:
            hardest_negatives.append(hit.docid)
            rank += 1
        if rank == results_limit:
            break
    return hardest_negatives

def scrub_tldrs(tldrs, positive_dataset_tags, datasets_file):
    datasets_list = json.load(open(datasets_file))
    out_rows = []
    for i, tldr in enumerate(tldrs):
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
    return out_rows

def convert_keyphrases(k):
    k = k.strip().capitalize()
    if k == "Training":
        k = "Training Style"
    elif k == "Length":
        k = "Text Length"
    elif k == "Language":
        k = "Language Required"
    return k

def construct_keyword_query(galactica_dict, keys):
    keywords = []
    for k in keys:
        if galactica_dict.get(k, None) is not None and galactica_dict.get(k, "").strip() not in ["", None, "None", "N/A"]:
            field = galactica_dict[k]
            if k == "length" and galactica_field.strip().lower() == "paragraph-level":
                galactica_field = "paragraph"
            elif k == "length" and galactica_field.strip().lower() == "sentence-level":
                galactica_field = "sentence"
            keywords.append(field.strip())
    keywords_raw = " ".join(keywords)
    final_keywords = ""
    for k in keywords_raw.lower().split():
        if k not in stopwords and k not in final_keywords:
            final_keywords = final_keywords + " " + k
    return final_keywords.strip()

if __name__ == "__main__":
    args = parser.parse_args()


    tagged_positives = list(load_rows(args.tagged_positives_file))
        
    galactica_parses = list(jsonlines.open(args.parse_file))
    raw_tldrs = [t["Single-Sentence Summary"] for t in galactica_parses]
    tldrs = scrub_tldrs(raw_tldrs, tagged_positives, args.datasets_file)
    tagged_negatives_pwc = load_rows_by_abstract(args.tagged_negatives_file)

    tagged_negatives = {}
    for k, row in tagged_negatives_pwc.items():
        tagged_negatives[k] = row

    if args.negative_mining == "hard":
        # Mine negatives via BM25 retrieval
        searcher = SimpleSearcher(args.anserini_index)
        # ^ Use default BM25 parameters for now
        # searcher.set_bm25(k1=0.8, b=0.4)

    raw_keyphrases = args.keyphrases_to_use.split(",")
    keyphrase_keys = [convert_keyphrases(k) for k in raw_keyphrases]

    merged_tags = []
    for i in tqdm(range(len(tagged_positives))):
        instance_abstract = tagged_positives[i]["abstract"]
        all_negatives = tagged_negatives[instance_abstract]["datasets"]
        merged_dict = tagged_positives[i]

        merged_dict["query"] = tldrs[i]
        keyphrase_query = construct_keyword_query(galactica_parses[i], keyphrase_keys)
        merged_dict["keyphrase_query"] = keyphrase_query

        merged_dict["positives"] = tagged_positives[i]["datasets"]

        if len(merged_dict["positives"]) == 0:
            continue
        del merged_dict["datasets"]
        if len(all_negatives) > args.num_negatives:
            if args.negative_mining == "sample":
                merged_dict["negatives"] = list(np.random.choice(all_negatives, size=args.num_negatives, replace=False))
            elif args.negative_mining == "hard":
                merged_dict["negatives"] = find_hard_negatives(searcher, tldrs[i], args.num_negatives, all_negatives)
            else:
                raise NotImplementedError
            if tldrs[i] == '':
                continue
            if len(merged_dict["negatives"]) == 0:
                continue
            assert len(merged_dict["negatives"]) > 0, breakpoint()

        merged_tags.append(merged_dict)
    write_rows(merged_tags, args.combined_file)