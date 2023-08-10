'''
python convert_tevatron_output_to_trec_eval.py \
    --output-trec-file REPO_PATH/tevatron_models/tevatron.trec \
    --tevatron-ranking REPO_PATH/tevatron_models/rank.tsv \
    --id2dataset REPO_PATH/tevatron_data/metadata/id2dataset.json \
    --test-queries REPO_PATH/tevatron_data/test_queries.jsonl \
    --search-collection dataset_search_collection.jsonl \
    --depth 5
'''

import argparse
from collections import defaultdict
import csv
import json
import jsonlines

parser = argparse.ArgumentParser()
parser.add_argument('--output-trec-file', type=str, default="REPO_PATH/tevatron_models/trec.output")
parser.add_argument('--tevatron-ranking', type=str, default="REPO_PATH/tevatron_models/rank.tsv")
parser.add_argument('--id2dataset', type=str, default="REPO_PATH/tevatron_data/metadata/id2dataset.json")
parser.add_argument('--test-queries', type=str, default="REPO_PATH/tevatron_data/test_queries.jsonl")
parser.add_argument('--search-collection', type=str, default="dataset_search_collection.jsonl")
parser.add_argument('--depth', type=int, required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    test_queries = list(jsonlines.open(args.test_queries))
    id2dataset = json.load(open(args.id2dataset))


    query_idx_to_year_mapping = {}
    for query_row in jsonlines.open(args.test_queries):
        query_idx_to_year_mapping[int(query_row["text_id"])] = query_row["year"]

    dataset_metadata = {}
    for row in jsonlines.open(args.search_collection):
        dataset_metadata[row["id"]] = row

    with open(args.output_trec_file, 'w') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(["QueryID", "Q0", "DocID", "Rank", "Score", "RunID"])

        query_datasets = defaultdict(set)

        previous_query = None
        current_query_retrieved_count = 0
        for row in open(args.tevatron_ranking).read().split("\n"):
            if len(row.split()) == 0:
                continue
            [query_idx, dataset_idx, score] = row.split()
            query_idx = int(query_idx)
            if query_idx != previous_query:
                current_query_retrieved_count = 0
                previous_query = query_idx
            query_year = query_idx_to_year_mapping[query_idx]
            raw_docid = id2dataset[dataset_idx]
            docid = "_".join(raw_docid.split())
            if docid in query_datasets[query_idx]:
                continue
            query_datasets[query_idx].add(docid)
            query = test_queries[query_idx]["query"]

            if "[SEP]" in query:
                query_id = "_".join(query.split("[SEP] ")[-1].split())
            else:
                query_id = "_".join(query.split())

            if dataset_metadata[raw_docid].get("year", None) is not None and query_year < dataset_metadata[raw_docid]["year"]:
                continue
            current_query_retrieved_count += 1
            if current_query_retrieved_count > args.depth:
                continue
            tsv_writer.writerow([query_id, "Q0", docid, str(len(query_datasets[query_idx])), score, "run-1"])
