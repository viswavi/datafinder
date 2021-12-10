# python data_analysis/dataset_exploration.py

import argparse
from collections import Counter
import json
import jsonlines
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--train-file', type=str, default="/projects/metis0_ssd/users/vijayv/dataset-recommendation/tagged_dataset_positives.jsonl")
parser.add_argument('--test-file', type=str, default="/projects/metis0_ssd/users/vijayv/dataset-recommendation/scirex_queries_and_datasets.json")
parser.add_argument('--datasets-file', type=str, default="/projects/metis0_ssd/users/vijayv/dataset-recommendation/datasets.json")

def construct_dataset_year_mapping(datasets_file):
    dataset_year_mapping = {}
    for dataset in json.load(open(datasets_file)):
        if dataset.get("introduced_date") is not None:
            year = int(dataset["introduced_date"].split('-')[0])
            dataset_year_mapping[dataset["name"]] = year
    return dataset_year_mapping

def dataset_counts(tags_object):
    datasets = []
    for row in tags_object:
        datasets.extend(row["datasets"])
    dataset_counts = Counter(datasets)
    rank_to_counts = []
    for i, (ds, count) in enumerate(dataset_counts.most_common()):
        rank_to_counts.append((i, count))
    rank_to_counts_str = " ".join([str(tup) for tup in rank_to_counts])
    print(f"rank_to_counts:\n{{{rank_to_counts_str}}}")
    print(f"zip(datasets, rank_to_counts):\n{{{list(zip(dict(dataset_counts.most_common()).keys(), rank_to_counts))}}}")

def datasets_by_venue(tags_object, venues_list, type):
    datasets = []
    num_papers_in_venues = 0
    years = []
    for row in tags_object:
        venue_match = row.get("venue") is not None and row["venue"] in venues_list
        journal_match = row.get("journal") is not None and row.get("journal") in venues_list
        if venue_match or journal_match:
            num_papers_in_venues += 1
            datasets.extend(row["datasets"])
            if type == "CV":
                years.append(row["year"])
    dataset_counts = Counter(datasets)
    return {k: round(float(count)/num_papers_in_venues * 100, 2) for (k, count) in dataset_counts.most_common()}

def datasets_by_field_analysis(positive_tags, venues, journals):
    top_venues = Counter(venues)
    top_venues = [v for v in top_venues if top_venues[v] >= 3 and v is not None]
    top_journals = Counter(journals)
    top_journals = [j for j in top_journals if top_journals[j] >= 3 and j is not None]
    
    CV_VENUES = []
    ML_VENUES = []
    NLP_VENUES = []
    ROBOTICS_VENUES = []
    SIGNAL_SPEECH_VENUES = []
    other_venues = []
    for v in top_venues + top_journals:
        if "CVPR" in v or "ICCV" in v or "WACV" in v or "Conference on Computer Vision and Pattern Recognition" in v or "International Conference on Computer Vision" in v or "Winter Conference on Applications of Computer Vision" in v:
            CV_VENUES.append(v)
        elif "NAACL" in v or "EMNLP" in v or "ACL" in v or "Empirical Methods in Natural Language Processing" in v or "Annual Meeting of the Association for Computational Linguistics" in v or "Transactions of the Association for Computational Linguistics" in v or "COLING" in v:
            # NACAL, EMNLP, ACL, TACL, COLING
            NLP_VENUES.append(v)
        elif "INTERSPEECH" in v or "Conference on Acoustics, Speech and Signal Processing" in v or "ICASSP" in v or "Automatic Speech Recognition and Understanding Workshop" in v:
            # Interspeech, ICASSP, ASRU
            SIGNAL_SPEECH_VENUES.append(v)
        elif "International Conference on Intelligent Robots and Systems" in v or "International Conference on Robotics and Automation" in v or "Robotics and Automation Letters" in v or "Journal of Robotics Research" in v:
            # IROS, ICRA, IJRR
            ROBOTICS_VENUES.append(v)
        elif "international conference on learning representations" in v.lower() or "ICLR" in v or "international conference on machine learning" in v.lower() or "ICML" in v or "neural information processing systems" in v.lower() or "neurips" in v.lower() or "NIPS" in v:
            # ICLR, ICML, NeurIPS
            ML_VENUES.append(v)
        else:
            other_venues.append(v)

    top_cv_datasets = datasets_by_venue(positive_tags, CV_VENUES, "CV")
    print(f"\nTop CV datasets:\n{top_cv_datasets}")
    top_nlp_datasets = datasets_by_venue(positive_tags, NLP_VENUES, "NLP")
    print(f"\nTop NLP datasets:\n{top_nlp_datasets}")
    top_speech_datasets = datasets_by_venue(positive_tags, SIGNAL_SPEECH_VENUES, "Signal/Speech")
    print(f"\nTop Signal Processing/Speech datasets:\n{top_speech_datasets}")
    top_ml_datasets = datasets_by_venue(positive_tags, ROBOTICS_VENUES, "Robotics")
    print(f"\nTop Robotics datasets:\n{top_ml_datasets}")
    top_ml_datasets = datasets_by_venue(positive_tags, ML_VENUES, "ML")
    print(f"\nTop ML datasets:\n{top_ml_datasets}")

def datasets_by_recency(positive_tags, dataset_year_mapping):
    recency = []
    for row in positive_tags:
        for dataset in row["datasets"]:
            if dataset in dataset_year_mapping:
                elapsed = row["year"] - dataset_year_mapping[dataset]
                if elapsed >= 0:
                    recency.append(elapsed)
    buckets = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 28)]
    frequencies_unordered = {}
    for elapsed in recency:
        for bucket in buckets:
            (bucket_start, bucket_end) = bucket
            if elapsed >= bucket_start and elapsed <= bucket_end:
                frequencies_unordered[bucket] = frequencies_unordered.get(bucket, 0) + 1
    frequencies = {}
    for bucket in sorted(frequencies_unordered.keys()):
        frequencies[bucket] = frequencies_unordered[bucket]
    print(f"frequencies: {frequencies}")
    print(f"np.median(recency): {np.median(recency)}")
    print(f"np.mean(recency): {np.mean(recency)}")

def datasets_used_frequencies(train_set_tags, test_set_tags):
    train_num_datasets = []
    for row in train_set_tags:
        train_num_datasets.append(len(row["datasets"]))
    test_num_datasets = []
    for row in test_set_tags:
        test_num_datasets.append(len(row["documents"]))
    train_counts = Counter(train_num_datasets).most_common()
    test_counts = Counter(test_num_datasets).most_common()
    relative_train_freq = {k: v/float(len(train_num_datasets)) for k, v in train_counts}
    relative_test_freq = {k: v/float(len(test_num_datasets)) for k, v in test_counts}
    breakpoint()

if __name__ == "__main__":
    args = parser.parse_args()
    train_tags_reader = jsonlines.open(args.train_file)
    train_tags = list(train_tags_reader)
    test_tags = json.load(open(args.test_file))

    # ds_counts = dataset_counts(positive_tags)
    journals = []
    venues = []
    for doc in train_tags:
        venues.append(doc["venue"])
        if "journal" in doc:
            journals.append(doc["journal"])

    datasets_by_field_analysis(train_tags, venues, journals)

    dataset_year_mapping = construct_dataset_year_mapping(args.datasets_file)
    datasets_by_recency(train_tags, dataset_year_mapping)

    datasets_used_frequencies(train_tags, test_tags)