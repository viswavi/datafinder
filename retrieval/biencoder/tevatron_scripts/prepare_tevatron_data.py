'''
python retrieval/biencoder/tevatron_scripts/prepare_tevatron_data.py \
    --tagged-datasets-file data/train_data.jsonl \
    --search-collection data/dataset_search_collection.jsonl \
    --test-queries data/test_data.jsonl \
    --output-training-directory tevatron_data/training \
    --output-search-directory tevatron_data/search \
    --output-query-file tevatron_data/test_queries.jsonl
'''

import argparse
import json
import jsonlines
import os
import sys
from tqdm import tqdm
sys.path.extend(["..", ".", "../..", "../../.."])

try:
    from data_processing.build_search_corpus.extract_methods_tasks_from_pwc import add_prompt_to_description, parse_tasks_from_evaluation_tables_file, parse_methods_from_methods_file
except:
    from extract_methods_tasks_from_pwc import add_prompt_to_description, parse_tasks_from_evaluation_tables_file, parse_methods_from_methods_file

parser = argparse.ArgumentParser()
parser.add_argument('--output-training-directory', type=str, default="tevatron_data/training_raw")
parser.add_argument('--output-search-directory', type=str, default="tevatron_data/search_raw")
parser.add_argument('--output-metadata-directory', type=str, default="tevatron_data/metadata")
parser.add_argument('--output-query-file', type=str, default="data/test_data.jsonl")
parser.add_argument('--num-shards', type=int, default=1, help="Number of shards of search collection to write")
parser.add_argument('--test-queries', type=str, default="scirex_queries_and_datasets.json")
parser.add_argument('--tagged-datasets-file', type=str, default="tagged_datasets_merged_random_negatives.jsonl")
parser.add_argument('--search-collection', type=str, default="dataset_search_collection.jsonl")
parser.add_argument('--evaluation-tables-file', type=str, default=None, help="Path to the evaluation-tables.json file")
parser.add_argument('--methods-file', type=str, default=None, help="Path to the methods.json file")

def format_search_text(row, separator=" [SEP] "):
    return separator.join([get_key_if_not_none(row, "contents"), get_key_if_not_none(row, "title"), get_key_if_not_none(row, "abstract")])

def generate_doc_ids(documents):
    '''
    Map datasets to integers, and write this bijection in both directions to disk.
    '''
    doc2idx = {}
    idx2doc = {}
    idx2text = {}
    for i, document in enumerate(documents):
        doc2idx[document['id']] = i
        idx2doc[i] = document['id']
        idx2text[i] = format_search_text(document)
    return doc2idx, idx2doc, idx2text

def load_rows(search_collection):
    return list(jsonlines.open(search_collection))

def write_rows(rows, outfile):
    with open(outfile, 'w') as f:
        writer = jsonlines.Writer(f)
        writer.write_all(rows)
    print(f"Wrote {len(rows)} rows to {outfile}.")

def generate_training_instances(training_set, doc2idx, idx2text, tasks=None, methods=None):
    '''
    Each line should look like
    {'query': TEXT_TYPE, 'positives': List[TEXT_TYPE], 'negatives': List[TEXT_TYPE]}
    '''
    training_rows = []
    for instance in tqdm(training_set):
        positives = [idx2text[doc2idx[pos]] for pos in instance["positives"]]
        negatives = [idx2text[doc2idx[neg]] for neg in instance["negatives"]]
        query = instance["tldr"]
        if tasks is not None and methods is not None:
            query = add_prompt_to_description(query, tasks, methods)
        training_rows.append({'query': query, "positives": positives, "negatives": negatives})
    return training_rows

def format_query_file(test_query_file, doc2idx, tasks=None, methods=None):
    '''
    Each line should look like
    {"text_id": xxx, "text": TEXT_TYPE}
    '''
    formatted_queries = []
    for i, row in tqdm(enumerate(json.load(open(test_query_file)))):
        dataset_idxs = [doc2idx[dataset] for dataset in row["documents"]]
        # formatted_queries.append({"text_id": dataset_idxs[0], "text": row["query"]})
        query = row["query"]
        if tasks is not None and methods is not None:
            query = add_prompt_to_description(query, tasks, methods)
        formatted_queries.append({"text_id": str(i), "text": query, "year": row["year"]})
    return formatted_queries

def get_key_if_not_none(map, key):
    value = map.get(key, None)
    if value is not None:
        value = value.replace('"', "'")
        return value
    else:
        return ""

def generate_inference_instances(raw_rows, doc2idx):
    '''
    Each line should look like
    {text_id: "xxx", 'text': TEXT_TYPE}
    '''
    search_rows = []
    for row in raw_rows:
        formatted = format_search_text(row)
        search_rows.append({"text_id": doc2idx[row["id"]], "text": formatted})
    return search_rows

if __name__ == "__main__":
    args = parser.parse_args()
    os.makedirs(args.output_training_directory, exist_ok=True)
    os.makedirs(args.output_search_directory, exist_ok=True)
    os.makedirs(args.output_metadata_directory, exist_ok=True)
    search_collection = load_rows(args.search_collection)
    dataset2id, id2dataset, id2text = generate_doc_ids(search_collection)

    if args.evaluation_tables_file is None or args.methods_file is None:
        tasks = None
        methods = None
    else:
        tasks = parse_tasks_from_evaluation_tables_file(args.evaluation_tables_file)
        methods = parse_methods_from_methods_file(args.methods_file)


    test_queries = format_query_file(args.test_queries, dataset2id, tasks, methods)
    write_rows(test_queries, os.path.join(args.output_query_file))

    search_rows = generate_inference_instances(search_collection, dataset2id)

    tagged_datasets = load_rows(args.tagged_datasets_file)
    training_rows = generate_training_instances(tagged_datasets, dataset2id, id2text, tasks, methods)
    write_rows(training_rows, os.path.join(args.output_training_directory, "train_data.json"))
    
    shards = []
    start_idx = 0
    for i in range(args.num_shards):
        end_idx = start_idx + float(len(search_rows))/args.num_shards
        if i == args.num_shards - 1:
            end_idx = len(search_rows)
        shards.append(search_rows[start_idx:end_idx])
        start_idx = end_idx

    for i, shard in enumerate(shards):
        write_rows(shard, os.path.join(args.output_search_directory, f"{i}.json"))

    dataset2id_path = os.path.join(args.output_metadata_directory, "dataset2id.json")
    json.dump(dataset2id, open(dataset2id_path, 'w'))
    json.dump(id2dataset, open(os.path.join(args.output_metadata_directory, "id2dataset.json"), 'w'))
    print(f"Wrote dataset2id file to {dataset2id_path}.")