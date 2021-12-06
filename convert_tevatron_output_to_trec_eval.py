'''
python convert_tevatron_output_to_trec_eval.py \
    --output-trec-file /projects/metis0_ssd/users/vijayv/dataset-recommendation/tevatron_models/tevatron.trec \
    --tevatron-ranking /projects/metis0_ssd/users/vijayv/dataset-recommendation/tevatron_models/rank.tsv \
    --id2dataset /projects/metis0_ssd/users/vijayv/dataset-recommendation/tevatron_data/metadata/id2dataset.json \
    --test-queries /projects/metis0_ssd/users/vijayv/dataset-recommendation/tevatron_data/test_queries.jsonl 
'''

import argparse
from collections import defaultdict
import csv
import json
import jsonlines

parser = argparse.ArgumentParser()
parser.add_argument('--output-trec-file', type=str, default="/projects/metis0_ssd/users/vijayv/dataset-recommendation/tevatron_models/trec.output")
parser.add_argument('--tevatron-ranking', type=str, default="/projects/metis0_ssd/users/vijayv/dataset-recommendation/tevatron_models/rank.tsv")
parser.add_argument('--id2dataset', type=str, default="/projects/metis0_ssd/users/vijayv/dataset-recommendation/tevatron_data/metadata/id2dataset.json")
parser.add_argument('--test-queries', type=str, default="/projects/metis0_ssd/users/vijayv/dataset-recommendation/tevatron_data/test_queries.jsonl ")


if __name__ == "__main__":
    args = parser.parse_args()
    test_queries = list(jsonlines.open(args.test_queries))
    id2dataset = json.load(open(args.id2dataset))

    with open(args.output_trec_file, 'w') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(["QueryID", "Q0", "DocID", "Rank", "Score", "RunID"])

        query_datasets = defaultdict(set)
        for row in open(args.tevatron_ranking).read().split("\n"):
            if len(row.split()) == 0:
                continue
            [query_idx, dataset_idx, score] = row.split()
            query_idx = int(query_idx)
            docid = "_".join(id2dataset[dataset_idx].split())
            if docid in query_datasets[query_idx]:
                continue
            query_datasets[query_idx].add(docid)
            query = test_queries[query_idx]["text"]

            if "[SEP]" in query:
                query_id = "_".join(query.split("[SEP] ")[-1].split())
            else:
                query_id = "_".join(query.split())

            tsv_writer.writerow([query_id, "Q0", docid, str(len(query_datasets[query_idx])), score, "run-1"])
