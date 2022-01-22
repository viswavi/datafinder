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
import csv
import json
import jsonlines
import numpy as np
import os
import sys
sys.path.extend(["..", ".", "../..", "../../.."])
from transformers import pipeline
from tqdm import tqdm
from utils import scrub_dataset_references

try:
    from data_processing.build_search_corpus.extract_methods_tasks_from_pwc import add_prompt_to_description, parse_tasks_from_evaluation_tables_file, parse_methods_from_methods_file
except:
    from extract_methods_tasks_from_pwc import add_prompt_to_description, parse_tasks_from_evaluation_tables_file, parse_methods_from_methods_file

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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
parser.add_argument('--task-method-masking-strategy', type=str, default=None, choices=['prompt', 'mask', None])
parser.add_argument('--use-keyphrases', action="store_true", required=False)
parser.add_argument('--test-set-query-annotations', type=str, default="test_set_query_annotations.csv")
parser.add_argument('--datasets-file', type=str, default="datasets.json", help="JSON file containing metadata about all datasets on PapersWithCode")

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

def generate_training_instances(training_set, doc2idx, idx2text, tasks=None, methods=None, use_keyphrases=True, keyphrases = [], method_strategy=None):
    '''
    Each line should look like
    {'query': TEXT_TYPE, 'positives': List[TEXT_TYPE], 'negatives': List[TEXT_TYPE]}
    '''
    training_rows = []
    for i, instance in tqdm(enumerate(training_set)):
        positives = [idx2text[doc2idx[pos]] for pos in instance["positives"]]
        negatives = [idx2text[doc2idx[neg]] for neg in instance["negatives"]]
        query = instance["tldr"]
        if tasks is not None and methods is not None:
            query = add_prompt_to_description(query, tasks, methods)
        if use_keyphrases is True:
            query = keyphrases[i]
        training_rows.append({'query': query, "positives": positives, "negatives": negatives})
    return training_rows

def format_query_file(test_query_file, doc2idx, tasks=None, methods=None, method_strategy=None):
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

def generate_keyphrases(tagged_datasets, batch_size=20, device=2):
    tokenizer = AutoTokenizer.from_pretrained("ankur310794/bart-base-keyphrase-generation-openkp")
    model = AutoModelForSeq2SeqLM.from_pretrained("ankur310794/bart-base-keyphrase-generation-openkp")
    generator = pipeline('text2text-generation', model=model, tokenizer=tokenizer, device=device)

    tldrs = []
    max_len = -1
    for instance in tagged_datasets:
        abstract = instance["abstract"]
        if len(abstract.split()) > max_len:
            max_len = len(abstract.split())
        if len(abstract.split()) > 700:
            abstract = " ".join(abstract.split()[:700])
        tldrs.append(abstract)
    keyphrases = []
    for batch_idx in tqdm(range(int(np.ceil(len(tldrs) / batch_size)))):
        outputs = generator(tldrs[batch_idx*batch_size:(batch_idx+1)*batch_size])
        kps = [output["generated_text"].replace(";", "") for output in outputs]
        keyphrases.extend(kps)
    return keyphrases


if __name__ == "__main__":
    args = parser.parse_args()
    os.makedirs(args.output_training_directory, exist_ok=True)
    os.makedirs(args.output_search_directory, exist_ok=True)
    os.makedirs(args.output_metadata_directory, exist_ok=True)
    search_collection = load_rows(args.search_collection)
    dataset2id, id2dataset, id2text = generate_doc_ids(search_collection)

    datasets_list = json.load(open(args.datasets_file))

if args.evaluation_tables_file is None or args.methods_file is None:
        tasks = None
        methods = None
    else:
        tasks = parse_tasks_from_evaluation_tables_file(args.evaluation_tables_file)
        methods = parse_methods_from_methods_file(args.methods_file)

    test_queries = format_query_file(args.test_queries, dataset2id, tasks, methods, method_strategy=args.task_method_masking_strategy)

    if args.use_keyphrases:
        query_keyword_mapping = {}
        for row in csv.DictReader(open(args.test_set_query_annotations)):
            description = row["Original Query"].replace("‽", ",")
            keywords = row["Keywords"].replace("‽", ",")
            if row["Bad query?"].lower().strip() == "yes":
                continue
            dataset_tags = json.loads(row["All Gold"].replace("‽", ",").replace("'", '"'))

            dataset_names = []
            for dataset in datasets_list:
                if dataset["name"] in dataset_tags:
                    dataset_names.append(dataset["name"])
                    # dataset_names.extend(dataset["variants"])
            dataset_names = list(set(dataset_names))
            tldr = scrub_dataset_references(description, dataset_names)
            query_keyword_mapping[tldr] = keywords
        json.dump([list(item) for item in query_keyword_mapping.items()], open(os.path.join(args.output_training_directory, "query_keyword_mapping.json"), 'w'))
        test_queries_replaced = []
        for query in test_queries:
            query["text"] = query_keyword_mapping[query["text"]]
            test_queries_replaced.append(query)
        test_queries = test_queries_replaced
    else:
        query_keyword_mapping = None

    write_rows(test_queries, os.path.join(args.output_query_file))


    search_rows = generate_inference_instances(search_collection, dataset2id)

    tagged_datasets = load_rows(args.tagged_datasets_file)
    if args.use_keyphrases:
        keyphrases = generate_keyphrases(tagged_datasets)
    else:
        keyphrases = []
    training_rows = generate_training_instances(tagged_datasets, dataset2id, id2text, tasks, methods, use_keyphrases=args.use_keyphrases, keyphrases=keyphrases, method_strategy=args.task_method_masking_strategy)
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