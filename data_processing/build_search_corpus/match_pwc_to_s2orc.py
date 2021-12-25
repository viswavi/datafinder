'''
python match_pwc_to_s2orc.py \
    --caches-directory s2orc_caches/ \
    --s2orc-metadata-directory S2ORC_PATH/s2orc_metadata/ \
    --pwc-datasets-file data/datasets.json \
    --pwc-to-s2orc-mapping-file pwc_dataset_to_s2orc_mapping.pkl
'''

import argparse
from collections import defaultdict
import json
import jsonlines
import os
import pickle
import time
from tqdm import tqdm

from paperswithcode import PapersWithCodeClient



parser = argparse.ArgumentParser()
parser.add_argument('--caches-directory', type=str, default="s2orc_caches/")
parser.add_argument('--s2orc-metadata-directory', type=str, default="S2ORC_PATH/s2orc_metadata/")
parser.add_argument('--pwc-datasets-file', type=str, default="data/datasets.json")
parser.add_argument('--pwc-to-s2orc-mapping', type=str, default="pwc_dataset_to_s2orc_mapping.pkl", help="Name of file within args.caches_directory")

class PwCMetadata:
    def __init__(self, title=None, doi=None, arxivId=None):
        self.title = title
        self.doi = doi
        self.arxivId = arxivId

    def __str__(self):
        return str({"title": str(self.title),
                    "doi": str(self.doi),
                    "arxivId": str(self.arxivId)}) 

def convert_metadata_to_field_dicts(dataset_metadata):
    title_dict = defaultdict(list)
    doi_dict = defaultdict(list)
    arxiv_id_dict = defaultdict(list)
    for dataset_name, meta in dataset_metadata.items():
        if meta.title != None:
            title_dict[meta.title.lower()].append(dataset_name)
        if meta.doi != None:
            doi_dict[meta.doi].append(dataset_name)
        if meta.arxivId != None:
            arxiv_id_dict[meta.arxivId].append(dataset_name)
    return title_dict, doi_dict, arxiv_id_dict

def main(s2orc_caches, metadata_directory, pwc_datasets_file, pwc_to_s2orc_mapping_file):
    start = time.perf_counter()
    os.makedirs(s2orc_caches, exist_ok=True)
    pwc_dataset_to_s2orc_mapping_file = os.path.join(s2orc_caches, pwc_to_s2orc_mapping_file)

    if os.path.exists(pwc_dataset_to_s2orc_mapping_file):
        dataset_to_s2orc_id_mapping = pickle.load(open(pwc_dataset_to_s2orc_mapping_file, 'rb'))
        print(f"Loaded {len(dataset_to_s2orc_id_mapping)} PwC dataset-to-S2orc mappings from  from {pwc_dataset_to_s2orc_mapping_file}.")
    else:
        dataset_meta_cache_file = os.path.join(s2orc_caches, "dataset_paper_meta_file.pkl")
        if os.path.exists(dataset_meta_cache_file):
            datasets_meta = pickle.load(open(dataset_meta_cache_file, 'rb'))
            print(f"Loaded {len(datasets_meta)} dataset metadata from {dataset_meta_cache_file}.")
        else:
            pwc_datasets = json.load(open(pwc_datasets_file))
            client = PapersWithCodeClient(token="b3693879190fdbbaeb0e79e92d6188e8e4ac5188")
            datasets_meta = {}
            missed_counter = 0
            api_call_failed = []
            for ds in tqdm(pwc_datasets):
                if ds["paper"] is None:
                    missed_counter += 1
                    continue
                paper_title = ds["paper"]["title"]
                url = ds["paper"]["url"]
                if "arxiv" in url:
                    arxiv_id = url.split("/")[-1]
                    arxiv_id = arxiv_id.split(".pdf")[0]
                    metadata = PwCMetadata(arxivId=arxiv_id, title=paper_title)
                elif "paperswithcode" in url:
                    pwc_id = url.split("https://paperswithcode.com/paper/")[1]
                    try:
                        pwc_meta = client.paper_get(pwc_id)
                        assert paper_title == pwc_meta.title, breakpoint()
                        metadata = PwCMetadata(arxivId=pwc_meta.arxiv_id,
                                            title=pwc_meta.title)
                    except:
                        api_call_failed.append(url)
                        continue
                elif "doi.org" in url:
                    doi = url.split("doi.org/")[-1]
                    metadata = PwCMetadata(doi=doi, title=paper_title)
                else:
                    missed_counter += 1
                datasets_meta[ds["name"]] = metadata

            print(f"{len(datasets_meta)} datasets linked to a paper. {missed_counter} datasets could not be linked.")
            api_calls_fmt = "\n".join(api_call_failed)
            print(f"{len(api_call_failed)} API calls failed:\n{api_calls_fmt}")
            pickle.dump(datasets_meta, open(dataset_meta_cache_file, 'wb'))

        # Initialize lookup dictionaries
        title_dict, doi_dict, arxiv_id_dict = convert_metadata_to_field_dicts(datasets_meta)

        dataset_to_s2orc_id_mapping = {}
        shard_files = [f for f in os.listdir(metadata_directory) if f.endswith(".jsonl")]
        for shard_file in tqdm(shard_files):
            for doc in jsonlines.open(os.path.join(metadata_directory, shard_file)):
                s2orc_id = doc["paper_id"]
                if doc["title"].lower() in title_dict:
                    matching_dataset_names = title_dict[doc["title"].lower()]
                    for dataset_name in matching_dataset_names:
                        if dataset_name in dataset_to_s2orc_id_mapping:
                            dataset_to_s2orc_id_mapping[dataset_name].append(s2orc_id)
                        dataset_to_s2orc_id_mapping[dataset_name] = [s2orc_id]
                    del title_dict[doc["title"].lower()]

                if doc["doi"] in doi_dict:
                    matching_dataset_names = doi_dict[doc["doi"]]
                    for dataset_name in matching_dataset_names:
                        if dataset_name in dataset_to_s2orc_id_mapping:
                            dataset_to_s2orc_id_mapping[dataset_name].append(s2orc_id)
                        dataset_to_s2orc_id_mapping[dataset_name] = [s2orc_id]
                    del doi_dict[doc["doi"]]

                if doc["arxiv_id"] in arxiv_id_dict:
                    matching_dataset_names = arxiv_id_dict[doc["arxiv_id"]]
                    for dataset_name in matching_dataset_names:
                        if dataset_name in dataset_to_s2orc_id_mapping:
                            dataset_to_s2orc_id_mapping[dataset_name].append(s2orc_id)
                        dataset_to_s2orc_id_mapping[dataset_name] = [s2orc_id]
                    del arxiv_id_dict[doc["arxiv_id"]]
            print(f"{len(dataset_to_s2orc_id_mapping)} datasets matched so far.")
        
        print("\n\n")

        print(f"{len(dataset_to_s2orc_id_mapping)} out of {len(datasets_meta)} paper-linked datasets were matched to S2orc.")
        pickle.dump(dataset_to_s2orc_id_mapping, open(pwc_dataset_to_s2orc_mapping_file, 'wb'))
        print(f"Wrote file to {pwc_dataset_to_s2orc_mapping_file}.")


    reversed_dataset_to_s2orc_mapping = defaultdict(list)
    for dataset_name, s2orc_ids in dataset_to_s2orc_id_mapping.items():
         for s2orc_id in s2orc_ids:
             reversed_dataset_to_s2orc_mapping[s2orc_id].append(dataset_name)

    shard_files = [f for f in os.listdir(metadata_directory) if f.endswith(".jsonl")]

    cited_datasets = defaultdict(list)
    for shard_file in tqdm(shard_files):
        for doc in jsonlines.open(os.path.join(metadata_directory, shard_file)):
            if doc.get('outbound_citations', []) == []:
                continue
            for outbound_citation in doc['outbound_citations']:
                if outbound_citation in reversed_dataset_to_s2orc_mapping:
                    cited_datasets[doc['paper_id']].extend(reversed_dataset_to_s2orc_mapping[outbound_citation])

    cited_datasets_file = os.path.join(s2orc_caches, "s2orc_papers_citing_datasets.json")
    print(f"{len(cited_datasets)} papers found that cite a dataset-paper. Wrote to {cited_datasets_file}")
    json.dump(cited_datasets, open(cited_datasets_file, 'w'))

    end = time.perf_counter()
    print(f"Script took {end - start} seconds.")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args.caches_directory, args.s2orc_metadata_directory, args.pwc_datasets_file, args.pwc_to_s2orc_mapping)