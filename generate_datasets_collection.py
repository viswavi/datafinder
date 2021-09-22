import argparse
import glob
import gzip
import json
import jsonlines
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--output-file', type=str, default="dataset_search_collection.jsonl")
parser.add_argument('--dataset-to-s2orc-mapping-file',
                    type=str,
                    default="/projects/ogma2/users/vijayv/extra_storage/s2orc_caches/pwc_dataset_to_s2orc_mapping.json",
                    help="This file contains the papers introducing each dataset")
parser.add_argument('--s2orc-full-text-directory', type=str, default="/projects/ogma2/users/vijayv/extra_storage/s2orc_caches/s2orc_full_texts/")
parser.add_argument('--s2orc-metadata-directory', type=str, default="/projects/ogma2/users/vijayv/extra_storage/s2orc_caches/s2orc_metadata/")

def load_metadata_iterator(metadata_files):
    for f in metadata_files:
        for row in jsonlines.open(f):
            yield row

def construct_dataset_document_collection(jsonl_writer, dataset_s2orc_id_mapping, s2orc_full_text_directory, s2orc_metadata_directory):

    dataset_metadata = {}
    for dataset_name, [s2orc_id] in dataset_s2orc_id_mapping.items():
        dataset_metadata[s2orc_id] = dataset_name
    document_search_list = set(dataset_metadata.keys())

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
            if doc['paper_id'] not in document_search_list:
                continue
            paper_id = doc['paper_id']

            dataset_document = {}
            dataset_document["dataset_name"] = dataset_metadata[paper_id]
            dataset_document["title"] = metadata[paper_id]["title"]
            dataset_document["abstract"] = metadata[paper_id]["abstract"]
            section_texts = [section["text"] for section in doc["body_text"]]
            body_text = "\n".join(section_texts)
            dataset_document["body_text"] = body_text
            jsonl_writer.write(dataset_document)
            document_search_list.remove(doc['paper_id'])
            if len(document_search_list) == 0:
                return

if __name__ == "__main__":
    args = parser.parse_args()
    dataset_s2orc_id_mapping = json.load(open(args.dataset_to_s2orc_mapping_file))
    with open(args.output_file, 'wb') as f:
        writer = jsonlines.Writer(f)
        construct_dataset_document_collection(writer, dataset_s2orc_id_mapping, args.s2orc_full_text_directory, args.s2orc_metadata_directory)