import argparse
import enchant
from collections import defaultdict
import glob
import gzip
import json
import jsonlines
import numpy as np
import os
import pickle
import random
import time
from tqdm import tqdm
from tokenizer import make_tok_seg

np.random.seed(1900)

# Construct spaCy sentence segmenter.
nlp = make_tok_seg()

en_dictionary = enchant.Dict("en_US")

# Directory containing S2ORC full texts (>75GB)
full_texts_directory = "/projects/ogma2/users/vijayv/extra_storage/s2orc_caches/s2orc_full_texts/"

def filter_variant(variant):
    if len(variant) <= 3:
        return False
    if en_dictionary.check(variant.lower()):
        return False
    if variant == "Kumar" or variant == "GoPro":
        return False
    return True

def load_introducing_paper_to_dataset_mapping():
    mapping = pickle.load(open("/projects/ogma2/users/vijayv/extra_storage/s2orc_caches/pwc_dataset_to_s2orc_mapping.pkl", 'rb'))
    reverse_mapping = {}
    for dataset, paperid in mapping.items():
        assert len(paperid) == 1
        reverse_mapping[paperid[0]] = dataset
    return reverse_mapping

def generate_candidate_snippets(num_snippets_to_label=-1, max_repetitions_for_dataset=5):
    pwc_datasets_file = "datasets.json"
    pwc_datasets = json.load(open(pwc_datasets_file))

    paper_to_dataset_mapping = load_introducing_paper_to_dataset_mapping()

    dataset_name_lookup_map = defaultdict(list)
    for dataset_meta in pwc_datasets:
        candidate_variants = [dataset_meta["name"]] + dataset_meta.get("variants", [])
        for variant in candidate_variants:
            if filter_variant(variant):
                dataset_name_lookup_map[dataset_meta["name"]].append(variant)
    
    full_texts = glob.glob(os.path.join(full_texts_directory, "*.jsonl.gz"))
    # Shuffle texts, in case there's some bias in the order of shards on S2ORC.
    np.random.shuffle(full_texts)

    candidate_dataset_counts = defaultdict(int)
    # List of datasets that have already been requested for labeling 
    capped_datasets = []

    candidate_snippets = []
    for shard_path in full_texts:
        shard_id = shard_path.split(".jsonl.gz")[0].split("/")[-1]
        print(f"Loaded shard {shard_id}.")

        dataset_json = os.path.join(full_texts_directory, f"paper_to_datasets_map_shard_{shard_id}.json")
        dataset_index = json.load(open(dataset_json))

        shard = gzip.open(shard_path, 'rt')
        s2orc_metadata = jsonlines.Reader(shard)
        hits = 0

        current_number_of_extracted_snippets = len(candidate_snippets)
        for doc in tqdm(s2orc_metadata):
            paper_id = doc['paper_id']
            s2_url=f"\n\nhttp://api.semanticscholar.org/CorpusID:{paper_id}"
            if paper_id not in dataset_index:
                continue
            dataset_hits = dataset_index[paper_id]

            dataset_bearing_section = None
            dataset_bearing_sentence = None
            variant = None

            dataset_introduced_by_paper = paper_to_dataset_mapping.get(paper_id, "")
            mention_hits = [dataset_name for hit_type, dataset_name in dataset_hits if hit_type == "mention"]
            if dataset_introduced_by_paper != "":
                # Ignore any datasets introduced by this paper, since this is less interesting for us to label.
                mention_hits = [dataset_name for dataset_name in mention_hits if dataset_name != dataset_introduced_by_paper]

            #reference_hits = [dataset_name for hit_type, dataset_name in dataset_hits if hit_type == "reference"]
            #mention_and_reference_hits = list(set(mention_hits).intersection(reference_hits))
            mention_and_reference_hits = list(set(mention_hits))

            mention_and_reference_hits = [ds for ds in mention_and_reference_hits if ds not in capped_datasets]
            if len(mention_and_reference_hits) == 0:
                continue

            # dataset_name = np.random.choice(mention_and_reference_hits)
            for dataset_name in mention_and_reference_hits:
                candidate_dataset_counts[dataset_name] += 1
                if candidate_dataset_counts[dataset_name] >= max_repetitions_for_dataset:
                    capped_datasets.append(dataset_name)

                dataset_variants = dataset_name_lookup_map[dataset_name]
                dataset_bearing_sentences = []

                critical_section=False
                for section in doc["body_text"]:
                    lowercased_section_title = section['section'].lower()
                    critical_section_headers = ["dataset", "experiment", "evaluation", "result", "training", "testing"]
                    noncritical_section_headers = ["related work", "future work", "discussion", "conclusion", "concluding"]

                    for header in noncritical_section_headers:
                        if header in lowercased_section_title:
                            critical_section=False
                            break
                    for header in critical_section_headers:
                        if header in lowercased_section_title:
                            critical_section=True
                            break

                    if not critical_section:
                        continue

                    section_text = section["text"]
                    variant_hits = [variant for variant in dataset_variants if variant in section_text]

                    if len(variant_hits) > 0:
                        sentences = list(nlp(section_text).sents)
                        for i, sentence in enumerate(sentences):
                            hit = False
                            for word in sentence:
                                s_word = str(word)
                                if s_word in variant_hits:
                                    dataset_bearing_section = section_text
                                    dataset_bearing_sentence = str(sentence)
                                    variant = s_word
                                    hit = True
                                    break
                            if hit:
                                sentence_context = ""
                                #if i > 0:
                                    #sentence_context += str(sentences[i-1]) + " "
                                sentence_context += str(sentences[i]) + " "
                                #if i < len(sentences) - 1:
                                    #sentence_context += str(sentences[i+1]) + " "
                                sentence_context = sentence_context[:-1]
                                dataset_bearing_sentences.append(sentence_context)

                if len(dataset_bearing_sentences) == 0:
                    continue

                candidate_snippets.append({
                                            "paper_id": paper_id,
                                            "candidate_dataset": dataset_name,
                                            "dataset_variants": dataset_variants,
                                            "candidate_sections": dataset_bearing_sentences,
                                        })
                if len(candidate_snippets) % 100 == 0:
                    print(f"{len(candidate_snippets)} snippets extracted.")
                    pickle.dump(candidate_snippets, open("dataset_sentence_snippets_all_mentions.pkl", 'wb'))

                if len(candidate_snippets) == num_snippets_to_label:
                    return candidate_snippets
        print(f"{len(candidate_snippets) - current_number_of_extracted_snippets} snippets extracted from shard {shard_id}.")

    return candidate_snippets


def main():
    candidate_snippets = generate_candidate_snippets()
    random.Random(0).shuffle(candidate_snippets)
    start = time.perf_counter()
    try:
        pickle.dump(candidate_snippets, open("dataset_sentence_snippets.pkl", 'wb'))
        end = time.perf_counter()
        print(f"Took {round(end - start, 2)} seconds.")
    except:
        breakpoint()

if __name__ == "__main__":
    main()