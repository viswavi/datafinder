import argparse
from collections import defaultdict
import glob
import gzip
import json
import jsonlines
import numpy as np
import os
import pickle
from tqdm import tqdm
from tokenizer import make_tok_seg

np.random.seed(1900)

# Construct spaCy sentence segmenter.
nlp = make_tok_seg()

parser = argparse.ArgumentParser()
parser.add_argument('--output-file', type=str, default="tagged_datasets.jsonl")
parser.add_argument('--s2orc-full-text-directory', type=str, default="/projects/ogma2/users/vijayv/extra_storage/s2orc_caches/s2orc_full_texts/")
parser.add_argument('--s2orc-metadata-directory', type=str, default="/projects/ogma2/users/vijayv/extra_storage/s2orc_caches/s2orc_metadata/")

def load_introducing_paper_to_dataset_mapping():
    mapping = pickle.load(open("/projects/ogma2/users/vijayv/extra_storage/s2orc_caches/pwc_dataset_to_s2orc_mapping.pkl", 'rb'))
    reverse_mapping = {}
    for dataset, paperid in mapping.items():
        assert len(paperid) == 1
        reverse_mapping[paperid[0]] = dataset
    return reverse_mapping

def tag_datasets(jsonl_writer, dataset_name_lookup_map, paper_to_dataset_mapping, full_texts_directory, metadata_directory):
    full_texts = glob.glob(os.path.join(full_texts_directory, "*.jsonl.gz"))
    # Shuffle texts, in case there's some bias in the order of shards on S2ORC.
    np.random.shuffle(full_texts)

    # List of datasets that have already been requested for labeling 
    num_documents_written = 0
    for full_text_shard_path in full_texts:
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

        for doc in tqdm(s2orc_full_text):
            paper_id = doc['paper_id']
            doc_metadata = metadata[paper_id]

            s2_url=f"\n\nhttp://api.semanticscholar.org/CorpusID:{paper_id}"
            if paper_id not in dataset_index:
                continue
            dataset_hits = dataset_index[paper_id]

            dataset_bearing_section = None
            dataset_bearing_sentence = None
            variant = None

            if paper_id in paper_to_dataset_mapping:
                # Ignore any dataset-introducing papers, since our focus is on system- or model-building papers.
                continue

            mention_hits = [dataset_name for hit_type, dataset_name in dataset_hits if hit_type == "mention"]

            reference_hits = [dataset_name for hit_type, dataset_name in dataset_hits if hit_type == "reference"]
            mention_and_reference_hits = list(set(mention_hits).intersection(reference_hits))

            if len(mention_and_reference_hits) == 0:
                continue

            dataset_tags = []
            # dataset_name = np.random.choice(mention_and_reference_hits)
            for dataset_name in mention_and_reference_hits:

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

                dataset_bearing_sentences_no_pretraining = []
                for s in dataset_bearing_sentences:
                    if "pretrain" not in s and "pre-train" not in s:
                        dataset_bearing_sentences_no_pretraining.append(s)

                if len(dataset_bearing_sentences_no_pretraining) >= 1:
                    dataset_tags.append(dataset_name)
    
            if len(dataset_tags) > 0:
                doc_metadata["datasets"] = dataset_tags
                jsonl_writer.write(doc_metadata)
                num_documents_written += 1
        print(f"{num_documents_written} documents written to file so far.")


def main():
    args = parser.parse_args()

    pwc_datasets_file = "datasets.json"
    pwc_datasets = json.load(open(pwc_datasets_file))
    dataset_name_lookup_map = defaultdict(list)
    for dataset_meta in pwc_datasets:
        candidate_variants = [dataset_meta["name"]] + dataset_meta.get("variants", [])
        for variant in candidate_variants:
            dataset_name_lookup_map[dataset_meta["name"]].append(variant)
    paper_to_dataset_mapping = load_introducing_paper_to_dataset_mapping()

    assert not os.path.exists(args.output_file), "Output file already exists"

    with open(args.output_file, 'wb') as f:
        writer = jsonlines.Writer(f)
        tag_datasets(writer, dataset_name_lookup_map, paper_to_dataset_mapping, args.s2orc_full_text_directory, args.s2orc_metadata_directory)

if __name__ == "__main__":
    main()