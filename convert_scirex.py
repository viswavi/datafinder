'''
Download SciREX data from https://github.com/allenai/SciREX/blob/master/scirex_dataset/release_data.tar.gz to
SCIREX_PATH/scirex_dataset/release_data/.

python convert_scirex.py \
    --scirex-directory $SCIREX_PATH/scirex_dataset/release_data/ \
    --dataset-search-collection dataset_search_collection.jsonl \
    --datasets-file datasets.json \
    --scirex-to-s2orc-metadata-file $PICKLES_DIRECTORY/scirex_id_to_s2orc_metadata_with_tldrs.pkl \
    --output-relevance-file data/test_dataset_collection.qrels \
    --output-queries-file test_queries.csv \
    --output-combined-file data/test_data.jsonl \
    --training-set-documents tagged_dataset_positives.jsonl \
    --bad-query-filter-map bad_tldrs_mapping.json
'''

import argparse
from collections import Counter
import csv
import json
import jsonlines
import os
import pickle
import string

from utils import scrub_dataset_references

def transformed_document(doc):
    return doc

def input_y_n():
    y_n = str(input())
    return y_n.strip() == "" or y_n.strip().lower() == "y"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scirex-directory', type=str, default="SCIREX_PATH/scirex_dataset/release_data/",
                        help="Path to local pickle file containing raw dataset sentences for labeling")
    parser.add_argument('--dataset-search-collection', type=str, default="dataset_search_collection.jsonl", help="Jsonlines file containing all the datasets in our \"search collection\"")
    parser.add_argument('--datasets-file', type=str, default="datasets.json", help="JSON file containing metadata about all datasets on PapersWithCode")
    parser.add_argument('--scirex-to-s2orc-metadata-file', type=str, help="Pickle file containing mapping (with tldrs) from SciREX paper IDs to S2ORC metadata", default="PICKLES_DIRECTORY/scirex_id_to_s2orc_metadata_with_tldrs.pkl")
    parser.add_argument('--output-relevance-file', type=str, default="data/test_dataset_collection.qrels")
    parser.add_argument('--output-queries-file', type=str, default="data/test_queries.json")
    parser.add_argument('--output-combined-file', type=str, default="data/test_data.json")
    parser.add_argument('--training-set-documents', type=str, default="tagged_datasets.jsonl")
    parser.add_argument('--bad-query-filter-map', type=str, help="Mapping for manual list of bad SciREX queries to filter out", default="bad_tldrs_mapping.json")

    args = parser.parse_args()
    
    dataset_search_collection = list(jsonlines.open(args.dataset_search_collection))
    variant_to_dataset_mapping = {}
    for dataset in sorted(dataset_search_collection, key = lambda x: len(x["contents"]), reverse=True):
        if len(dataset["id"].split()) == 0:
            continue
        variant_to_dataset_mapping[dataset["id"]] = dataset["id"]
        for var in dataset["variants"]:
            if len(var.split()) == 0:
                continue
            variant_to_dataset_mapping[var] = dataset["id"]

    scirex_to_s2orc_metadata = pickle.load(open(args.scirex_to_s2orc_metadata_file, 'rb'))
    relevance_file = open(args.output_relevance_file, 'w')
    tsv_writer = csv.writer(relevance_file, delimiter='\t')
    tsv_writer.writerow(["QueryID", "0", "DocID", "Relevance"])
    query_writer = open(args.output_queries_file, 'w')

    training_set_tags = list(jsonlines.open(args.training_set_documents))
    tagged_datasets = [tag for document in training_set_tags for tag in document["datasets"]]
    tag_counts = Counter(tagged_datasets)
    rare_datasets = set()
    cumulative_count = 0
    least_common_dataset_counts = tag_counts.most_common()[-int(0.8 * len(tag_counts)):]
    for doc, count in least_common_dataset_counts:
        rare_datasets.add(doc)
        cumulative_count += count

    datasets_list = json.load(open(args.datasets_file))
    bad_query_filter_map = json.load(open(args.bad_query_filter_map))

    print(f"{round(len(rare_datasets) / float(len(tag_counts)) * 100, 2)}% most rare datasets make up only {round(cumulative_count/ float(len(tagged_datasets)) * 100, 2)}% of mentions")

    mismatches_cache = {}

    documents_containing_rare_datasets = 0
    total_rare_datasets_in_test_set = 0
    num_rows_written = 0

    queries_and_datasets = []
    for file in os.listdir(args.scirex_directory):
        if file.startswith(".") or not file.endswith(".jsonl"):
            continue
        scirex_split_file = os.path.join(args.scirex_directory, file)
        for doc in jsonlines.open(scirex_split_file):
            converted_doc = {}
            converted_doc["paper_id"] = doc["doc_id"]
            first_paragraph = " ".join(doc["words"][doc["sections"][0][0]:doc["sections"][0][1]])
            paragraphs = []
            for section in doc["sections"]:
                paragraphs.append(" ".join(doc["words"][section[0]:section[1]]))
            first_paragraph = paragraphs[0]
            body_text = "\n".join(paragraphs)
            converted_doc["abstract"] = first_paragraph
            converted_doc["body_text"] = body_text
            datasets_labeled = list(set([relation["Material"] for relation in doc["n_ary_relations"]]))
            dataset_mentions = []
            for dataset in datasets_labeled:
                single_dataset_mentions = []
                for mention_span in doc["coref"][dataset]:
                    mention_start, mention_end = mention_span
                    mention = " ".join(doc["words"][mention_start:mention_end])
                    single_dataset_mentions.append(mention)
                dataset_mentions.append(single_dataset_mentions)

            dataset_tags = []
            for ds, mentions in zip(datasets_labeled, dataset_mentions):
                dataset_matched = False
                ds_reformatted_1 = ds.replace("_", " ")
                ds_reformatted_2 = ds.replace("_", "-")
                ds_reformatted_3 = ds.replace("_", "")
                mention_candidates = [ds, ds_reformatted_1, ds_reformatted_2, ds_reformatted_3] + mentions
                for mention in mention_candidates:
                    matched = False
                    matched_variant = ""
                    if mention in variant_to_dataset_mapping:
                        matched = True
                        matched_variant = mention
                    if not matched:
                        for variant in variant_to_dataset_mapping:
                            if mention in variant:
                                matched = True
                                matched_variant = variant
                                break
                    if matched:
                        dataset_tags.append(variant_to_dataset_mapping[matched_variant])
                        dataset_matched = True
                        print(f"\nmention {matched_variant} matched\n")
                        break
                if not dataset_matched:
                    print(f"dataset {ds} with mentions {mention_candidates} omitted")
                    continue
            dataset_tags = list(set(dataset_tags))
            if len(dataset_tags) == 0:
                continue
            if len(set(dataset_tags).intersection(rare_datasets)) > 0:
                documents_containing_rare_datasets += 1
                total_rare_datasets_in_test_set += len(set(dataset_tags).intersection(rare_datasets))
            if doc["doc_id"] not in scirex_to_s2orc_metadata:
                continue
            tldr = scirex_to_s2orc_metadata[doc["doc_id"]]["tldr"]
            if "None <|TLDR|>" in tldr:
                continue

            assert tldr in bad_query_filter_map
            if bad_query_filter_map[tldr] is True:
                continue

            dataset_names = []
            for dataset in datasets_list:
                if dataset["name"] in dataset_tags:
                    dataset_names.append(dataset["name"])
                    # dataset_names.extend(dataset["variants"])
            dataset_names = list(set(dataset_names))
            tldr = scrub_dataset_references(tldr, dataset_names)
            if len(set(dataset_tags)) != len(set(datasets_labeled)):
                print(f"Possible dataset mismatch found: {list(set(dataset_tags))} vs {list(set(datasets_labeled))}")
                print("Is this mismatch ok? Y/n")
                mismatch_accepted = True
                '''
                mismatch_accepted = input_y_n()
                mismatches_cache[tuple([tuple(set(dataset_tags)), tuple(set(datasets_labeled))])] = mismatch_accepted
                '''
                if not mismatch_accepted:
                    continue
            query_writer.write(tldr + "\n")
            year = scirex_to_s2orc_metadata[doc["doc_id"]]["year"]
            query_id = "_".join(tldr.split())
            for dataset in dataset_tags:
                docid = "_".join(dataset.split())
                tsv_writer.writerow([query_id, "Q0", docid, "1"])
            num_rows_written += 1

            row = {"tldr": tldr, "positives": sorted(dataset_tags), "year": year}
            queries_and_datasets.append(row)
    query_writer.close()
    relevance_file.close()

    print(f"Wrote {num_rows_written} test documents to {args.output_relevance_file}")

    with open(args.output_combined_file, 'w') as outfile:
        writer = jsonlines.Writer(outfile)
        writer.write_all(queries_and_datasets)
    print(f"Wrote {num_rows_written} test documents to {args.output_combined_file}")

    print(f"{documents_containing_rare_datasets} test set documents contain rare datasets, giving an average of {round(float(total_rare_datasets_in_test_set) / documents_containing_rare_datasets, 2)} rare datasets per test set document.")
    # pickle.dump(mismatches_cache, open("dataset_mismatches_cache.pkl", 'wb'))

    # Fields:
    # 'paper_id', 'title', 'authors', 'abstract', 'year'

    # main(scirex_documents, range_to_label, args.out_labels_directory, range_string)