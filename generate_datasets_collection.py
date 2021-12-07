'''
python generate_datasets_collection.py

python generate_datasets_collection.py --exclude-abstract --exclude-full-text --output-file dataset_search_collection_no_abstracts_or_paper_text.jsonl
python generate_datasets_collection.py --exclude-abstract --output-file dataset_search_collection_no_paper_text.jsonl

This script converts the full set of datasets from PapersWithCode into a jsonlines file, read to be consumed into an
Anserini search index.

Each row must have at least two columns: "id" (dataset name) and "contents" (description), among other fields.
'''

import argparse
import glob
import gzip
import json
import jsonlines
import os
import requests
import time
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--output-file', type=str, default="dataset_search_collection.jsonl")
parser.add_argument('--pwc-datasets-file', type=str, default="datasets.json")
parser.add_argument('--dataset-to-s2orc-mapping-file',
                    type=str,
                    default="/projects/ogma2/users/vijayv/extra_storage/s2orc_caches/pwc_dataset_to_s2orc_mapping.json",
                    help="This file contains the papers introducing each dataset")
parser.add_argument('--s2orc-full-text-directory', type=str, default="/projects/ogma2/users/vijayv/extra_storage/s2orc_caches/s2orc_full_texts/")
parser.add_argument('--s2orc-metadata-directory', type=str, default="/projects/ogma2/users/vijayv/extra_storage/s2orc_caches/s2orc_metadata/")
parser.add_argument('--exclude-abstract', action="store_true")
parser.add_argument('--exclude-full-text', action="store_true")


def load_metadata_iterator(metadata_files):
    for f in metadata_files:
        for row in jsonlines.open(f):
            yield row

def construct_dataset_to_description_mapping(pwc_datasets_metadata):
    dataset_description_mapping = {}
    for dataset in pwc_datasets_metadata:
        dataset_description_mapping[dataset["name"]] = dataset.get("description", "")
    return dataset_description_mapping

def construct_paper_info_to_dataset_mapping(pwc_datasets_metadata):
    paper_title_to_datasets_map = {}
    paper_url_to_datasets_map = {}
    for dataset in pwc_datasets_metadata:
        paper_meta = dataset.get("paper")
        if paper_meta is None:
            paper_meta = {}
        dataset_paper_title = paper_meta.get("title")
        dataset_url = paper_meta.get("url")
        if dataset_paper_title is not None:
            paper_title_to_datasets_map[dataset_paper_title] = dataset["name"]
        if dataset_url is not None:
            paper_url_to_datasets_map[dataset_url] = dataset["name"]
    return paper_title_to_datasets_map, paper_url_to_datasets_map

def construct_dataset_to_variants_mapping(pwc_datasets_metadata):
    dataset_variants_mapping = {}
    for dataset in pwc_datasets_metadata:
        dataset_variants_mapping[dataset["name"]] = dataset.get("variants", "")
    return dataset_variants_mapping

def construct_dataset_to_date_mapping(pwc_datasets_metadata):
    dataset_date_mapping = {}
    for dataset in pwc_datasets_metadata:
        dataset_date_mapping[dataset["name"]] = dataset.get("introduced_date", None)
    return dataset_date_mapping

def construct_dataset_document_collection(jsonl_writer,
                                          dataset_s2orc_id_mapping,
                                          pwc_datasets_metadata,
                                          s2orc_full_text_directory,
                                          s2orc_metadata_directory,
                                          exclude_abstract,
                                          exclude_full_text):
    dataset_to_descriptions = construct_dataset_to_description_mapping(pwc_datasets_metadata)
    dataset_to_variants = construct_dataset_to_variants_mapping(pwc_datasets_metadata)
    dataset_to_date = construct_dataset_to_date_mapping(pwc_datasets_metadata)
    paper_title_to_datasets_map, paper_url_to_datasets_map = construct_paper_info_to_dataset_mapping(pwc_datasets_metadata)
    dataset_to_paper_title_map = {dataset:title for title, dataset in paper_title_to_datasets_map.items()}

    s2orc_id_to_dataset_mapping = {}
    s2orc_ids_search_list = []
    for dataset_name, [s2orc_id] in dataset_s2orc_id_mapping.items():
        s2orc_id_to_dataset_mapping[s2orc_id] = dataset_name
        s2orc_ids_search_list.append(s2orc_id)

    matched_datasets = set()
    full_texts_files = glob.glob(os.path.join(s2orc_full_text_directory, "*.jsonl.gz"))
    for full_text_shard_path in full_texts_files:
        shard_id = full_text_shard_path.split(".jsonl.gz")[0].split("/")[-1]
        print(f"Loaded shard {shard_id}.")

        metadata_shard_path = os.path.join(s2orc_metadata_directory, f"{shard_id}.jsonl")
        s2orc_metadata = jsonlines.open(metadata_shard_path)
        metadata = {}
        for row in s2orc_metadata:
            metadata[row["paper_id"]] = row

        full_text_shard = gzip.open(full_text_shard_path, 'rt')
        s2orc_full_text = jsonlines.Reader(full_text_shard)
        for doc in tqdm(s2orc_full_text):
            dataset_name = None
            paper_id_match = False
            paper_title_match = False
            paper_id = doc['paper_id']

            if paper_id in s2orc_ids_search_list:
                paper_id_match = True
                dataset_name = s2orc_id_to_dataset_mapping.get(paper_id)
                s2orc_ids_search_list.remove(paper_id)

            paper_title = metadata[paper_id].get("title")
            if paper_title in paper_title_to_datasets_map:
                paper_title_match = True
                if dataset_name is None:
                    dataset_name = paper_title_to_datasets_map.get(paper_title)
                del paper_title_to_datasets_map[paper_title]

            if not (paper_id_match or paper_title_match):
                continue
            assert dataset_name is not None

            dataset_document = {}
            dataset_document["id"] = dataset_name
            dataset_document["contents"] = dataset_to_descriptions[dataset_name]
            dataset_document["variants"] = dataset_to_variants[dataset_name]
            dataset_document["title"] = metadata[paper_id]["title"]
            dataset_document["abstract"] = metadata[paper_id]["abstract"]
            dataset_document["year"] = metadata[paper_id]["year"]
            if dataset_name in dataset_to_date and dataset_to_date[dataset_name] is not None:
                dataset_document["date"] = dataset_to_date[dataset_name]
            section_texts = [section["text"] for section in doc["body_text"]]
            body_text = "\n".join(section_texts)
            dataset_document["body_text"] = body_text

            matched_datasets.add(dataset_name)
            if exclude_abstract:
                del dataset_document["abstract"]
            if exclude_full_text:
                del dataset_document["body_text"]
            jsonl_writer.write(dataset_document)

        print(f"{len(matched_datasets)} documents written to file so far.")

    num_matched_datasets = len(matched_datasets)
    # Write unmatched datasets
    for dataset_name in dataset_to_descriptions:
        if dataset_name not in matched_datasets:
            dataset_document = {}
            dataset_document["id"] = dataset_name
            dataset_title = dataset_to_paper_title_map.get(dataset_name)
            if dataset_title is not None:
                dataset_document["title"] = dataset_title
            else:
                dataset_document["title"] = ""
            dataset_document["contents"] = dataset_to_descriptions[dataset_name]
            dataset_document["variants"] = dataset_to_variants[dataset_name]
            if dataset_name in dataset_to_date and dataset_to_date[dataset_name] is not None:
                dataset_document["date"] = dataset_to_date[dataset_name]
                dataset_document["year"] = int(dataset_to_date[dataset_name].split('-')[0])
            if not exclude_abstract:
                dataset_document["abstract"] = ""
            if not exclude_full_text:
                dataset_document["body_text"] = ""
            matched_datasets.add(dataset_name)
            jsonl_writer.write(dataset_document)
    print(f"Wrote {len(matched_datasets) - num_matched_datasets} datasets that could not be matched to S2orc.")


if __name__ == "__main__":
    args = parser.parse_args()
    dataset_s2orc_id_mapping = json.load(open(args.dataset_to_s2orc_mapping_file))
    with open(args.pwc_datasets_file, 'rb') as f:
        pwc_datasets_metadata = json.load(f)
    dataset_descriptions = construct_dataset_to_description_mapping(pwc_datasets_metadata)
    dataset_variants = construct_dataset_to_variants_mapping(pwc_datasets_metadata)

    with open(args.output_file, 'wb') as f:
        writer = jsonlines.Writer(f)
        construct_dataset_document_collection(writer,
                                              dataset_s2orc_id_mapping,
                                              pwc_datasets_metadata,
                                              args.s2orc_full_text_directory,
                                              args.s2orc_metadata_directory,
                                              args.exclude_abstract,
                                              args.exclude_full_text)