'''
python evaluate_rare_dataset_recall.py \
/projects/ogma1/vijayv/dataset-recommendation/data/test/test_dataset_collection.qrels \
/projects/ogma1/vijayv/dataset-recommendation/data/test/retrieved_documents_knn_weighted.trec
'''

import argparse
from collections import defaultdict, Counter
import csv
import jsonlines
import numpy as np

def construct_rare_dataset_list(training_set_tags):
    tagged_datasets = [tag for document in training_set_tags for tag in document["datasets"]]
    tag_counts = Counter(tagged_datasets)
    rare_datasets = set()
    cumulative_count = 0
    least_common_dataset_counts = tag_counts.most_common()[-int(0.8 * len(tag_counts)):]
    for dataset, count in least_common_dataset_counts:
        rare_datasets.add("_".join(dataset.split()))
        cumulative_count += count
    return rare_datasets

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
    assert relevant.keys() == retrieved.keys()
    recalls = []
    for query in relevant:
        relevant_rare_datasets = [r for r in relevant[query] if r in rare_datasets]
        if len(relevant_rare_datasets) == 0:
            continue
        recall_k = len(relevant[query]) # Since we're doing R-Precision-style evaluation
        retrieved_rare_datasets = [r for r in retrieved[query][:recall_k] if r in rare_datasets]
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
    rare_datasets = construct_rare_dataset_list(training_set_tags)
    rare_dataset_recall = compute_average_rare_dataset_recall(relevant_csv, retrieved_csv, rare_datasets)
    print(f"rare_dataset_recall: {rare_dataset_recall}")