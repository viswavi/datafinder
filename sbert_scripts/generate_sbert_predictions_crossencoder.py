'''
python sbert_scripts/generate_sbert_predictions_crossencoder.py \
    --crossencoder-model-directory sbert_models/bert_hard_negatives \
    --biencoder-model-directory sbert_models/bert_hard_negatives \
    --biencoder_query_reps /projects/metis0_ssd/users/vijayv/dataset-recommendation/tevatron_data/test_queries_hard_negatives_longer_input_scibert/query.pt \
    --biencoder_passage_reps /projects/metis0_ssd/users/vijayv/dataset-recommendation/tevatron_data/search_encoded_hard_negatives_longer_input_scibert/*.pt \
    --search-collection dataset_search_collection.jsonl \
    --test-queries tevatron_data/test_queries.jsonl  \
    --output-file sbert_models/bert_hard_negatives/sbert.trec \
    --batch-size 300 \
    --results-limit 5
'''

import argparse
import glob
import jsonlines
from itertools import chain
import numpy as np
import os
import sys
from tqdm import tqdm
import torch

from sentence_transformers.cross_encoder import CrossEncoder

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from generate_knn_results import knn_search, write_hits_to_tsv
from generate_sbert_predictions_biencoder import construct_search_index
from prepare_tevatron_data import format_search_text

from tevatron.faiss_retriever.retriever import BaseFaissIPRetriever

parser = argparse.ArgumentParser()
parser.add_argument('--crossencoder-model-directory', type=str, default="sbert_models/bert_hard_negatives")
parser.add_argument("--search-collection", type=str, default="dataset_search_collection.jsonl", help="Test collection of queries and documents")
parser.add_argument('--test-queries', type=str, default="test_queries.jsonl", help="List of newline-delimited queries")
parser.add_argument('--output-file', type=str, default="sbert_models/bert_hard_negatives/sbert.trec", help="Retrieval file, in TREC format")
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--first-stage-depth', type=int, default=10)
parser.add_argument('--first-stage-batch-size', type=int, default=128)
parser.add_argument('--results-limit', type=int, default=5)
parser.add_argument('--biencoder_query_reps', type=str, default="/projects/metis0_ssd/users/vijayv/dataset-recommendation/tevatron_data/test_queries_hard_negatives_longer_input_scibert/0.pt")
parser.add_argument('--biencoder_passage_reps', type=str, default="/projects/metis0_ssd/users/vijayv/dataset-recommendation/tevatron_data/search_encoded_hard_negatives_longer_input_scibert/*.pt")

def read_dataset_collection(search_collection):
    dataset_texts = []
    dataset_ids = []
    for dataset_row in jsonlines.open(search_collection):
        dataset_texts.append(format_search_text(dataset_row))
        dataset_ids.append(dataset_row["id"])
    return dataset_texts, dataset_ids

def search_queries_biencoder(retriever, q_reps, p_lookup, args, depth=100, batch_size=128):
    if args.batch_size > 0:
        all_scores, all_indices = retriever.batch_search(q_reps, depth, batch_size)
    else:
        all_scores, all_indices = retriever.search(q_reps, depth)

    psg_indices = [[str(p_lookup[x]) for x in q_dd] for q_dd in all_indices]
    psg_indices = np.array(psg_indices)
    return all_scores, psg_indices

def do_first_stage_retrieval(query_reps, passage_reps, depth=100, batch_size=128):
    index_files = glob.glob(passage_reps)
    p_reps_0, p_lookup_0 = torch.load(index_files[0])
    retriever = BaseFaissIPRetriever(p_reps_0.float().numpy())

    shards = chain([(p_reps_0, p_lookup_0)], map(torch.load, index_files[1:]))
    if len(index_files) > 1:
        shards = tqdm(shards, desc='Loading shards into index', total=len(index_files))
    look_up = []
    for p_reps, p_lookup in shards:
        retriever.add(p_reps.float().numpy())
        look_up += p_lookup

    q_reps, q_lookup = torch.load(query_reps)
    q_reps = q_reps.float().numpy()

    all_scores, psg_indices = search_queries_biencoder(retriever, q_reps, look_up, args, depth=depth, batch_size=batch_size)
    return all_scores, psg_indices

if __name__ == "__main__":
    args = parser.parse_args()

    all_scores, psg_indices = do_first_stage_retrieval(args.query_reps, args.passage_reps, depth=args.first_stage_depth, batch_size=args.first_stage_batch_size)



    model = CrossEncoder(args.crossencoder_model_directory, num_labels=1, max_length=512)

    query_texts = []
    for row in jsonlines.open(args.test_queries):
        query_texts.append(row["text"])

    dataset_texts, dataset_ids = read_dataset_collection(args.search_collection)
    
    all_query_dataset_pairs = []
    for i, query in enumerate(query_texts):
        first_stage_doc_idxs = [int(docidx) for docidx in psg_indices[i]]
        first_stage_dataset_texts = [dataset_texts[docidx] for docidx in first_stage_doc_idxs]
        for dataset_text in first_stage_dataset_texts:
            all_query_dataset_pairs.append([query, dataset_text])

    scores = model.predict(all_query_dataset_pairs, batch_size=args.batch_size, show_progress_bar=True)

    all_ranks = []
    all_scores = []
    for i in range(len(query_texts)):
        first_stage_doc_idxs = [int(docidx) for docidx in psg_indices[i]]

        query_predictions_start = args.first_stage_depth * i
        query_predictions_end = args.first_stage_depth * (i+1)
        query_scores = -1 * scores[query_predictions_start:query_predictions_end]
        query_reranks = np.argsort(query_scores)[:args.results_limit]
        all_ranks.append([first_stage_doc_idxs[rerank] for rerank in query_reranks])
        all_scores.append(list(query_scores[query_reranks]))
    all_hits = knn_search(all_scores, all_ranks, dataset_ids)
    write_hits_to_tsv(args.output_file, all_hits, query_texts, args.results_limit)