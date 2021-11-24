'''
python prepare_reranker_data.py \
    --tagged-datasets-file tagged_datasets_merged_hard_negatives.jsonl \
    --search-collection dataset_search_collection.jsonl \
    --tokenizer-name allenai/scibert_scivocab_uncased \
    --output-training-data reranker_data/training_data_merged_negatives.json \
    --output-qid2query reranker_data/qid2query.json
'''

import argparse
from collections import defaultdict
import json
import jsonlines
import os
from transformers import AutoTokenizer

from prepare_tevatron_data import generate_doc_ids, load_rows

parser = argparse.ArgumentParser()
parser.add_argument('--tagged-datasets-file', required=True, type=str, default="tagged_datasets_merged_hard_negatives.jsonl")
parser.add_argument('--search-collection', required=True, type=str, default="dataset_search_collection.jsonl")
parser.add_argument('--tokenizer-name', required=True, type=str, default="allenai/scibert_scivocab_uncased")
parser.add_argument('--output-training-data', required=True, type=str, default="reranker_data/training_data_merged_negatives.json")
parser.add_argument('--output-qid2query', required=True, type=str, default="reranker_data/qid2query.json")
parser.add_argument('--max-length', type=int, default=512)

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
        qid = instance["paper_id"]
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

    with open(args.output_training_data, 'w') as f:
        for qid in queries.keys():
            item_set = {
                'qry': queries[qid],
                'pos': list(query_2_pos_docs[qid].values()),
                'neg': query_2_neg_docs[qid],
            }
            f.write(json.dumps(item_set) + '\n')


    with open(args.output_qid2query, 'w') as f:
        json.dump(qid2query, f)