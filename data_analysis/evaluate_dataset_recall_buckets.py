'''
python evaluate_rare_dataset_recall.py \
REPO_PATH/data/test/test_dataset_collection.qrels \
REPO_PATH/data/test/retrieved_documents_knn_weighted.trec
'''

import argparse
from collections import defaultdict, Counter
import csv
import jsonlines
import numpy as np

def construct_quintile_dataset_lists(training_set_tags, training_frequency_buckets):
    bucket_lists = []
    tagged_datasets = [tag for document in training_set_tags for tag in document["datasets"]]
    tag_counts = Counter(tagged_datasets)

    for frequency_bucket in training_frequency_buckets:
        bucket_start, bucket_end = frequency_bucket
        datasets_in_bucket = set()
        for dataset, frequency in tag_counts.most_common():
            if frequency >= bucket_start and frequency <= bucket_end:
                datasets_in_bucket.add("_".join(dataset.split()))
        bucket_lists.append(list(datasets_in_bucket))
    return bucket_lists

def read_relevant_csv(f):
    queries = defaultdict(list)
    with open(f) as csvfile:
        csvreader = csv.reader(csvfile, delimiter="\t")
        header = True
        for row in csvreader:
            if header:
                header = False
                continue
            [query_id, _, docid, score] = row
            if float(score) >= 1:
                queries[query_id].append(docid)
    return queries

def read_retrieved_csv(f):
    queries = defaultdict(list)
    with open(f) as csvfile:
        csvreader = csv.reader(csvfile, delimiter="\t")
        header = True
        for row in csvreader:
            if header:
                header = False
                continue
            [query_id, _, docid, _, score, _] = row
            queries[query_id].append(docid)
    return queries

def compute_average_rare_dataset_recall(relevant, retrieved, rare_datasets):
    recalls = []
    for query in relevant:
        retrieved_results = retrieved.get(query, [])
        relevant_rare_datasets = [r for r in relevant[query] if r in rare_datasets]
        if len(relevant_rare_datasets) == 0:
            continue
        recall_k = len(relevant[query]) # Since we're doing R-Precision-style evaluation
        retrieved_rare_datasets = [r for r in retrieved_results[:recall_k] if r in rare_datasets]
        relevant_retrieved_rare_datasets = set(retrieved_rare_datasets).intersection(relevant_rare_datasets)
        recall = float(len(relevant_retrieved_rare_datasets)) / float(len(relevant_rare_datasets))
        recalls.append(recall)
    return np.mean(recalls)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('relevant_trec_file', type=str)
    parser.add_argument('retrieved_trec_file', type=str)
    parser.add_argument('--training-set-documents', type=str, default="tagged_dataset_positives.jsonl")
    args = parser.parse_args()

    relevant_csv = read_relevant_csv(args.relevant_trec_file)
    retrieved_csv = read_retrieved_csv(args.retrieved_trec_file)

    training_set_tags = list(jsonlines.open(args.training_set_documents))

    training_frequency_buckets = [(501, 2500), (101, 500), (51, 100), (21, 50), (6, 20), (1, 5)]
    all_bucket_datasets = construct_quintile_dataset_lists(training_set_tags, training_frequency_buckets)
    for bucket_idx, bucket_datasets in enumerate(all_bucket_datasets):
        rare_dataset_recall = compute_average_rare_dataset_recall(relevant_csv, retrieved_csv, bucket_datasets)
        print(f"Recall in bucket {training_frequency_buckets[bucket_idx]}: {rare_dataset_recall}")