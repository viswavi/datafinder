'''
python sbert_scripts/generate_sbert_predictions_biencoder.py \
    --model-directory sbert_models/bert_hard_negatives\
    --search-collection dataset_search_collection.jsonl \
    --test-queries tevatron_data/test_queries.jsonl  \
    --output-file sbert_models/bert_hard_negatives/sbert.trec \
    --results-limit 5
'''

import argparse
import jsonlines
import numpy as np
import os
import sys

import faiss
from sentence_transformers import SentenceTransformer

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from generate_knn_results import knn_search, write_hits_to_tsv
from prepare_tevatron_data import format_search_text

parser = argparse.ArgumentParser()
parser.add_argument('--model-directory', type=str, default="sbert_models/bert_hard_negatives")
parser.add_argument("--search-collection", type=str, default="dataset_search_collection.jsonl", help="Test collection of queries and documents")
parser.add_argument('--test-queries', type=str, default="test_queries.jsonl", help="List of newline-delimited queries")
parser.add_argument('--output-file', type=str, default="sbert_models/bert_hard_negatives/sbert.trec", help="Retrieval file, in TREC format")
parser.add_argument('--results-limit', type=int, default=5)

def read_dataset_collection(search_collection):
    dataset_texts = []
    dataset_ids = []
    for dataset_row in jsonlines.open(search_collection):
        dataset_texts.append(format_search_text(dataset_row))
        dataset_ids.append(dataset_row["id"])
    return dataset_texts, dataset_ids

if __name__ == "__main__":
    args = parser.parse_args()

    model = SentenceTransformer(args.model_directory)

    query_texts = []
    for row in jsonlines.open(args.test_queries):
        query_texts.append(row["text"])

    dataset_texts, dataset_ids = read_dataset_collection(args.search_collection)
    
    all_query_dataset_pairs = []
    for query in query_texts:
        for dataset_text in dataset_texts:
            all_query_dataset_pairs.append([query, dataset_text])

    scores = model.predict(all_query_dataset_pairs)

    all_ranks = []
    all_scores = []
    for i, s in enumerate(scores):
        query_predictions_start = len(query_texts) * i
        query_predictions_end = len(query_texts) * (i+1)
        query_scores = scores[query_predictions_start:query_predictions_end]
        query_ranks = np.argsort(query_scores)[:args.results_limit]
        all_ranks.append(list(query_ranks))
        all_scores.append(list(query_scores[query_ranks]))
    all_hits = knn_search(all_scores, all_ranks, dataset_ids)
    write_hits_to_tsv(args.output_file, all_hits, query_texts, args.results_limit)