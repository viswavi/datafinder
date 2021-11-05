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

def construct_search_index(search_collection, model):
    dataset_texts = []
    dataset_ids = []
    for dataset_row in jsonlines.open(search_collection):
        dataset_texts.append(format_search_text(dataset_row))
        dataset_ids.append(dataset_row["id"])
    dataset_encodings = model.encode(dataset_texts)
    vectors = np.array(dataset_encodings, dtype=np.float32)
    index = faiss.IndexFlatL2(vectors.shape[1])
    # index = faiss.GpuIndexFlatL2(vectors.shape[0])
    index.add(vectors)
    return index, dataset_ids

if __name__ == "__main__":
    args = parser.parse_args()

    model = SentenceTransformer(args.model_directory)

    query_texts = []
    for row in jsonlines.open(args.test_queries):
        query_texts.append(row["text"])
    query_encodings = model.encode(query_texts)

    faiss_index, dataset_ids = construct_search_index(args.search_collection, model)
    knn_distances, knn_indices = faiss_index.search(query_encodings, args.results_limit)

    all_hits = knn_search(knn_distances, knn_indices, dataset_ids)
    write_hits_to_tsv(args.output_file, all_hits, query_texts, args.results_limit)