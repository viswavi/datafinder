'''
python prepare_reranker_data.py \
    --tagged-datasets-file tagged_datasets_merged_hard_negatives.jsonl \
    --search-collection dataset_search_collection.jsonl \
    --tokenizer-name allenai/scibert_scivocab_uncased \
    --output-dir reranker_data \
    --output-training-data-prefix data \
    --output-qid2query qid2query.json
'''

import argparse
from collections import defaultdict
import json
import os
import random
import sys
from transformers import AutoTokenizer
sys.path.extend(["..", ".", "../.."])

from retrieval.biencoder.tevatron_scripts.prepare_tevatron_data import generate_doc_ids, load_rows

parser = argparse.ArgumentParser()
parser.add_argument('--tagged-datasets-file', required=True, type=str, default="tagged_datasets_merged_hard_negatives.jsonl")
parser.add_argument('--search-collection', required=True, type=str, default="dataset_search_collection.jsonl")
parser.add_argument('--tokenizer-name', required=True, type=str, default="allenai/scibert_scivocab_uncased")
parser.add_argument('--output-dir', required=True, type=str, default="reranker_data")
parser.add_argument('--output-training-data-prefix', required=True, type=str, default="data")
parser.add_argument('--output-qid2query', required=True, type=str, default="qid2query.json")
parser.add_argument('--max-length', type=int, default=512)
parser.add_argument('--seed', type=int, default=2021)
parser.add_argument('--validation-split', type=float, default=0.1)

def generate_training_instances(training_set, doc2idx, idx2text, max_length):
    '''
    Each line should look like
    qid, qry, pos_id, pos, neg_id, neg
    '''
    query_2_pos_docs = defaultdict(dict)
    query_2_neg_docs = defaultdict(list)

    qid2query = {}
    queries = {}

    tokenize = lambda text: tokenizer.encode(text, add_special_tokens=False, max_length=max_length, truncation=True)

    for instance in training_set:
        qry = instance["tldr"]
        qid = "paper_id_" + instance["paper_id"]
        qid2query[qid] = qry

        query_dict = {
            'qid': qid,
            'query': tokenize(qry),
        }
        queries[qid] = query_dict

        positives = instance["positives"]
        for pos_id in positives:
            pos_text = idx2text[doc2idx[pos_id]]
            pos_tok = tokenize(pos_text)
            pos_dict = {
                'pid': pos_id,
                'passage': pos_tok,
            }
            query_2_pos_docs[qid][pos_id] = pos_dict

        negatives = instance["negatives"]
        for neg_id in negatives:
            neg_text = idx2text[doc2idx[neg_id]]
            neg_tok = tokenize(neg_text)
            neg_dict = {
                'pid': neg_id,
                'passage': neg_tok,
            }
            query_2_neg_docs[qid].append(neg_dict)

    return query_2_pos_docs, query_2_neg_docs, queries, qid2query

if __name__ == "__main__":
    args = parser.parse_args()
    search_collection = load_rows(args.search_collection)
    dataset2id, id2dataset, id2text = generate_doc_ids(search_collection)
    tagged_datasets = load_rows(args.tagged_datasets_file)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)

    query_2_pos_docs, query_2_neg_docs, queries, qid2query = generate_training_instances(tagged_datasets, dataset2id, id2dataset, args.max_length)

    query_ids = list(queries.keys())
    rng = random.Random(args.seed)
    train_query_ids = rng.sample(query_ids, k=int(len(query_ids) * args.validation_split))
    validation_query_ids = [k for k in query_ids if k not in train_query_ids]
    rng.shuffle(validation_query_ids)
    split_query_ids = [train_query_ids, validation_query_ids]

    for split_idx, split_name in enumerate(["train", "dev"]):
        os.makedirs(os.path.join(args.output_dir, split_name), exist_ok=True)
        split_file = os.path.join(args.output_dir, split_name, args.output_training_data_prefix + "_" + split_name + ".json")
        with open(split_file, 'w') as f:
            for qid in split_query_ids[split_idx]:
                item_set = {
                    'qry': queries[qid],
                    'pos': list(query_2_pos_docs[qid].values()),
                    'neg': query_2_neg_docs[qid],
                }
                f.write(json.dumps(item_set) + '\n')
        with open(os.path.join(args.output_dir, args.output_qid2query), 'w') as f:
            json.dump(qid2query, f)