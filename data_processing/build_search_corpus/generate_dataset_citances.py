'''
python generate_dataset_citances.py
'''

import argparse
from collections import defaultdict
import glob
import gzip
import json
import jsonlines
import os
import requests
import spacy
from spacy.pipeline.senter import DEFAULT_SENTER_MODEL

import time
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--datasets-to-citances-file', type=str, default="STORAGE_PATH/datasets_to_citances.json")
parser.add_argument('--pwc-datasets-file', type=str, default="datasets.json")
parser.add_argument('--dataset-to-s2orc-mapping-file',
                    type=str,
                    default="STORAGE_PATH/s2orc_caches/pwc_dataset_to_s2orc_mapping.json",
                    help="This file contains the papers introducing each dataset")
parser.add_argument('--citance-context-window', type=int, default=1)
parser.add_argument('--s2orc-full-text-directory', type=str, default="STORAGE_PATH/s2orc_caches/s2orc_full_texts/")
parser.add_argument('--s2orc-metadata-directory', type=str, default="STORAGE_PATH/s2orc_caches/s2orc_metadata/")

nlp = spacy.load("en_core_web_sm")

def match_ref_span_to_sentence_idx(ref_span, sentence_boundaries):
    start_idx = ref_span["start"]
    end_idx = ref_span["end"]
    sentence_idx = -1
    for sentence_idx, (start, end) in enumerate(sentence_boundaries):
        if start_idx >= start and end_idx <= end:
            break
    return sentence_idx 

def find_citances(dataset_to_s2orc_id_mapping, s2orc_id_to_dataset_names, full_texts_directory, metadata_directory, citance_context_window=1):
    full_texts = glob.glob(os.path.join(full_texts_directory, "*.jsonl.gz"))
    # Shuffle texts, in case there's some bias in the order of shards on S2ORC.
    # List of datasets that have already been requested for labeling 
    dataset_s2orc_ids = set(s2orc_id_to_dataset_names.keys())

    dataset_citances = {}
    for full_text_shard_path in tqdm(full_texts):
        shard_id = full_text_shard_path.split(".jsonl.gz")[0].split("/")[-1]
        print(f"Loaded shard {shard_id}.")

        dataset_json = os.path.join(full_texts_directory, f"paper_to_datasets_map_shard_{shard_id}.json")
        dataset_index = json.load(open(dataset_json))

        full_text_shard = gzip.open(full_text_shard_path, 'rt')
        s2orc_full_text = jsonlines.Reader(full_text_shard)

        metadata_shard_path = os.path.join(metadata_directory, f"{shard_id}.jsonl")

        s2orc_metadata = jsonlines.open(metadata_shard_path)
        metadata = {}
        for row in s2orc_metadata:
            metadata[row["paper_id"]] = row

        for doc in s2orc_full_text:
            paper_id = doc['paper_id']
            doc_metadata = metadata[paper_id]
            cited_dataset_paper_ids = list(set(doc_metadata.get("outbound_citations", [])).intersection(dataset_s2orc_ids))
            if len(cited_dataset_paper_ids) == 0:
                continue
            if len(doc.get('body_text', [])) == 0:
                continue
            for cited_dataset_paper_id in cited_dataset_paper_ids:
                variant_names = s2orc_id_to_dataset_names[cited_dataset_paper_id]
                canonical_dataset_name = variant_names[0] # Based on the way I ordered the variant names list when populating it.
                if canonical_dataset_name not in dataset_citances:
                    dataset_citances[canonical_dataset_name] = {}
                matching_bibref = None
                for bibref_id, bibref in doc.get("bib_entries", {}).items():
                    if bibref["link"] == cited_dataset_paper_id:
                        matching_bibref = bibref_id
                        break
                if matching_bibref == None:
                    continue
                for paragraph in doc.get('body_text', []):
                    matching_bib = None
                    for cite in paragraph['cite_spans']:
                        if cite["ref_id"] == matching_bibref:
                            matching_bib = cite
                            break
                    if matching_bib is not None:
                        variant_name_found = False
                        for variant in variant_names:
                            if variant in paragraph["text"]:
                                variant_name_found = True
                                break
                        # if variant_name_found:
                        #     breakpoint()

                        sentences = list(nlp(paragraph["text"]).sents)
                        sentence_boundaries = [(sbounds.start_char, sbounds.end_char) for sbounds in sentences]
                        sentence_idx = match_ref_span_to_sentence_idx(matching_bib, sentence_boundaries)
                        
                        window_start = max(sentence_idx - citance_context_window, 0)
                        window_end = min(sentence_idx + citance_context_window, len(sentences) - 1)
                        sentences_window = [sentences[sent_idx].text for sent_idx in range(window_start, window_end + 1)]
                        sentences_window_concat = " ".join(sentences_window)
                        
                        citance_example = {
                            "citing_paper_s2orc_id": paper_id,
                            "citance_paragraph": paragraph["text"],
                            "truncated_window": sentences_window_concat,
                            "num_sentences_in_citance": len(sentences_window),
                            "dataset_mentioned_by_name": variant_name_found
                        }
                        if paper_id not in dataset_citances[canonical_dataset_name]:
                            dataset_citances[canonical_dataset_name][paper_id] = []
                        dataset_citances[canonical_dataset_name][paper_id].append(citance_example)

    return dataset_citances



def main():
    args = parser.parse_args()

    pwc_datasets = list(json.load(open(args.pwc_datasets_file)))
    dataset_to_s2orc_mapping = json.load(open(args.dataset_to_s2orc_mapping_file))
    dataset_to_s2orc_id_mapping = {}
    s2orc_id_to_dataset_mapping = {}
    for dataset_name, [s2orc_id] in dataset_to_s2orc_mapping.items():
        dataset_to_s2orc_id_mapping[dataset_name] = s2orc_id
        s2orc_id_to_dataset_mapping[s2orc_id] = [dataset_name]
        matching_dataset_entry = None
        for dataset_entry in pwc_datasets:
            if dataset_entry["name"] == dataset_name:
                matching_dataset_entry = dataset_entry
        new_variants = list(set(matching_dataset_entry.get('variants', [])).difference(set([dataset_name])))
        s2orc_id_to_dataset_mapping[s2orc_id].extend(new_variants)

    dataset_citances = find_citances(dataset_to_s2orc_id_mapping, s2orc_id_to_dataset_mapping, args.s2orc_full_text_directory, args.s2orc_metadata_directory)
    assert not os.path.exists(args.datasets_to_citances_file), "Output file already exists"
    json.dump(dataset_citances, open(args.datasets_to_citances_file, 'w'))

if __name__ == "__main__":
    main()