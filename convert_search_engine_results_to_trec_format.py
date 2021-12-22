'''
python convert_search_engine_results_to_trec_format.py \
    pwc_search_results.tsv \
    pwc_search_results.trec \
    --depth 5

python convert_search_engine_results_to_trec_format.py \
    google_search_results.tsv \
    google_search_results.trec \
    --depth 5
'''

import argparse
from collections import defaultdict
import csv
import json
import jsonlines
from tqdm import tqdm
from utils import scrub_dataset_references


parser = argparse.ArgumentParser()
parser.add_argument('search_engines_file', type=str, help='CSV file containing results from a search engine (Google Datasets or PwC)')
parser.add_argument('output_trec_file', type=str, help='Output in TREC format')
parser.add_argument('--depth', type=int, default=5, help="Max results to return per query")
parser.add_argument('--test-set-annotations', type=str, default="data/test_data.jsonl")
parser.add_argument('--datasets-file', type=str, default="datasets.json", help="JSON file containing metadata about all datasets on PapersWithCode")

def interleave_results(all_query_results, max_results):
    interleaved_results = []
    i = 0
    each_remaining = [True for _ in all_query_results]
    while True:
        if sum(each_remaining) == 0 or len(interleaved_results) == max_results:
            break
        if len(all_query_results[i]) == 0:
            each_remaining[i] = False
        else:
            next_result = all_query_results[i][0]
            if next_result not in interleaved_results:
                interleaved_results.append(next_result)
            all_query_results[i] = all_query_results[i][1:]
        if i == len(all_query_results)-1:
            i = 0
        else:
            i += 1
    return interleaved_results

if __name__ == "__main__":
    args = parser.parse_args()
    out_rows = []

    datasets_list = json.load(open(args.datasets_file))
    datasets = {}
    for row in datasets_list:
        datasets[row["name"]] = row

    test_set_annotations = jsonlines.open(args.test_set_annotations)
    query_metadata = {}
    for row in test_set_annotations:
        query_metadata[row["tldr"]] = row

    unique_queries = []
    combined_query_results = defaultdict(list)

    for row in open(args.search_engines_file).readlines():
        cols = row.split('\t')
        query = cols[0]
        dataset_names = []
        tagged_datasets = json.loads(cols[2])
        for dataset in datasets_list:
            if dataset["name"] in tagged_datasets:
                dataset_names.append(dataset["name"])
                # dataset_names.extend(dataset["variants"])
        dataset_names = list(set(dataset_names))

        query_scrubbed = scrub_dataset_references(query, dataset_names)
        if query_scrubbed not in query_metadata:
            continue
        if query_scrubbed not in unique_queries:
            unique_queries.append(query_scrubbed)

        results = []
        for r in cols[2:]:
            if r not in datasets:
                continue
            dataset_date = datasets[r].get("introduced_date")
            if dataset_date is not None:
                dataset_year = int(dataset_date.split('-')[0])
                if query_metadata[query_scrubbed]["year"] < dataset_year:
                    continue
            results.append(r)
        combined_query_results[query_scrubbed].append(results)

    with open(args.output_trec_file, 'w') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(["QueryID", "Q0", "DocID", "Rank", "Score", "RunID"])

        for query, results in combined_query_results.items():
            query_id = "_".join(query.split())
            interleaved_results = interleave_results(results, args.depth)
            for i, doc in enumerate(interleaved_results):
                score = float(args.depth) - i
                docid = "_".join(doc.split())
                tsv_writer.writerow([query_id, "Q0", docid, str(i+1), score, "run-1"])