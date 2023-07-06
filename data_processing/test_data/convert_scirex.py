'''
python convert_scirex.py \
    --scirex-directory /projects/ogma2/users/vijayv/extra_storage/SciREX/scirex_dataset/release_data \
    --dataset-search-collection dataset_search_collection.jsonl \
    --annotations-file /projects/ogma1/vijayv/dataset-recommendation/aggregated_form_results.json \
    --datasets-file datasets.json \
    --scirex-to-s2orc-metadata-file /home/vijayv/pickle_backups/scirex_id_to_s2orc_metadata_with_tldrs.pkl \
    --output-relevance-file test_dataset_collection.qrels \
    --output-queries-file test_queries.csv \
    --output-multiqueries-file test_multiqueries.jsonl \
    --output-combined-file scirex_queries_and_datasets.json 
'''

import argparse
import csv
import itertools
import json
import jsonlines
import os
import nltk
import numpy as np
import pickle
import time

from collections import Counter

stopwords = nltk.corpus.stopwords.words("english")

def transformed_document(doc):
    return doc

def input_y_n():
    y_n = str(input())
    return not y_n.strip().lower() == "n"


def input_csv(options):
    valid_input = False
    comma_separated_values = []
    while not valid_input:
        valid_input = True
        csv_input = str(input())
        if len(csv_input.strip()) == 0:
            return []
        comma_separated_values = [c.strip() for c in csv_input.split(",")]
        for c in comma_separated_values:
            if c not in options:
                valid_input = False
                print(f"Dataset {c} is not in our dataset index. Try again:\n\n")
                break
    return comma_separated_values

def convert_keyphrases(k):
    k = k.strip().capitalize()
    if k == "Training":
        k = "Training Style"
    elif k == "Length":
        k = "Text Length"
    elif k == "Language":
        k = "Language Required"
    return k

def construct_keyword_query(galactica_dict, keys):
    keywords = []
    for k in keys:
        if galactica_dict.get(k, None) is not None and galactica_dict.get(k, "").strip() not in ["", None, "None", "N/A"]:
            keywords.append(galactica_dict[k].strip())
    keywords_raw = " ".join(keywords)
    final_keywords = "" 
    for k in keywords_raw.lower().split():
        if not k in stopwords and k not in final_keywords:
            final_keywords = final_keywords + " " + k
    return final_keywords.strip()

def filter_keywords(search_string):
    tokens = search_string.split()
    filtered_tokens = []
    for t in tokens:
        remove = False
        for prev in filtered_tokens:
            if t in prev or prev in t:
                remove = True
        if t in stopwords:
            remove = True
        if not remove:
            filtered_tokens.append(t)
    return " ".join(filtered_tokens)

def construct_multiple_keyword_queries(galactica_dict, keys):
    multi_keywords = []
    for k in keys:
        if galactica_dict.get(k, None) is not None and galactica_dict.get(k, "").strip() not in ["", None, "None", "N/A"]:
            galactica_field = galactica_dict[k]
            if k == "text_length" and galactica_field.strip().lower() == "paragraph-level":
                galactica_field = "paragraph"
            elif k == "text_length" and galactica_field.strip().lower() == "sentence-level":
                galactica_field = "sentence"
            multi_keywords.append(galactica_field.strip().split(','))

    multiple_searches = []
    for product_keywords in itertools.product(*multi_keywords):
        multiple_searches.append(filter_keywords(" ".join(product_keywords).lower()))

    return list(set(multiple_searches))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scirex-directory', type=str, default="/projects/ogma2/users/vijayv/extra_storage/SciREX/scirex_dataset/release_data",
                        help="Path to local pickle file containing raw dataset sentences for labeling")
    parser.add_argument('--dataset-search-collection', type=str, default="dataset_search_collection.jsonl", help="Jsonlines file containing all the datasets in our \"search collection\"")
    parser.add_argument('--annotations-file', type=str, default="/projects/ogma1/vijayv/dataset-recommendation/aggregated_form_results.json", help="File containing human annotations of datasets")
    parser.add_argument('--datasets-file', type=str, default="datasets.json", help="JSON file containing metadata about all datasets on PapersWithCode")
    parser.add_argument('--scirex-to-s2orc-metadata-file', type=str, help="Pickle file containing mapping (with tldrs) from SciREX paper IDs to S2ORC metadata", default="/home/vijayv/pickle_backups/scirex_id_to_s2orc_metadata_with_tldrs.pkl")
    parser.add_argument('--output-relevance-file', type=str, default="test_dataset_collection.qrels")
    parser.add_argument('--output-queries-file', type=str, default="test_queries.csv")
    parser.add_argument('--output-multiqueries-file', type=str, default="test_multiqueries.jsonl")
    parser.add_argument('--output-combined-file', type=str, default="scirex_queries_and_datasets.json")
    parser.add_argument('--training-set-documents', type=str, default="tagged_datasets.jsonl")
    parser.add_argument('--use-keyphrases', action="store_true")
    parser.add_argument('--keyphrases-to-use', type=str, default="task,modality,domain,language,length")
    parser.add_argument('--keep-queries-with-field', type=str, default=None)

    args = parser.parse_args()
    
    if args.use_keyphrases:
        keyphrase_keys = args.keyphrases_to_use.split(",")
    else:
        keyphrase_keys = "task,modality,domain,language,length".split(',')

    backup_file = args.output_combined_file + "l"

    backup_files_by_abstract = {}
    if os.path.exists("scirex_queries_and_datasets.jsonl"):
        backup_files = list(jsonlines.open("scirex_queries_and_datasets.jsonl", 'r'))
        for f in backup_files:
            backup_files_by_abstract[f["abstract"]] = f["documents"]

    backup_writer = jsonlines.Writer(open(backup_file, 'w'), flush=True)

    annotations = json.load(open(args.annotations_file))
    abstract_to_annotations = {}
    for annotation in annotations:
        abstract_to_annotations[annotation["abstract"]] = annotation

    dataset_search_collection = list(jsonlines.open(args.dataset_search_collection))
    variant_to_dataset_mapping = {}
    dataset_ids = []
    for dataset in sorted(dataset_search_collection, key = lambda x: len(x["contents"]), reverse=True):
        if len(dataset["id"].split()) == 0:
            continue
        dataset_ids.append(dataset["id"])
        variant_to_dataset_mapping[dataset["id"]] = dataset["id"]
        for var in dataset["variants"]:
            if len(var.split()) == 0:
                continue
            variant_to_dataset_mapping[var] = dataset["id"]

    scirex_to_s2orc_metadata = pickle.load(open(args.scirex_to_s2orc_metadata_file, 'rb'))
    tlds = [scirex_to_s2orc_metadata[row]["tldr"] for row in scirex_to_s2orc_metadata]
    relevance_file = open(args.output_relevance_file, 'w')
    tsv_writer = csv.writer(relevance_file, delimiter='\t')
    tsv_writer.writerow(["QueryID", "0", "DocID", "Relevance"])
    query_writer = open(args.output_queries_file, 'w')

    multiquery_rows = []

    mismatches_cache = {}

    counter = 0
    num_rows_written = 0

    no_datasets_tagged_counter = 0

    precisions = []
    recalls = []
    exact_match = []

    queries_and_datasets = []
    for file in os.listdir(args.scirex_directory):
        if file.startswith(".") or not file.endswith(".jsonl"):
            continue
        scirex_split_file = os.path.join(args.scirex_directory, file)
        for doc in jsonlines.open(scirex_split_file):
            counter += 1
            first_paragraph = " ".join(doc["words"][doc["sections"][0][0]:doc["sections"][0][1]])
            paragraphs = []
            for section in doc["sections"]:
                paragraphs.append(" ".join(doc["words"][section[0]:section[1]]))
            first_paragraph = paragraphs[0]
            body_text = "\n".join(paragraphs)
            datasets_labeled = list(set([relation["Material"] for relation in doc["n_ary_relations"]]))
            dataset_mentions = []
            for dataset in datasets_labeled:
                single_dataset_mentions = []
                for mention_span in doc["coref"][dataset]:
                    mention_start, mention_end = mention_span
                    mention = " ".join(doc["words"][mention_start:mention_end])
                    single_dataset_mentions.append(mention)
                dataset_mentions.append(single_dataset_mentions)

            matched_variants = []
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
                        matched_variants.append(matched_variant)
                        dataset_matched = True
                        # print(f"\nmention {matched_variant} matched\n")
                        break
                if not dataset_matched:
                    pass
                    #print(f"dataset {ds} with mentions {mention_candidates} omitted")
                    continue
            dataset_tags = list(set(dataset_tags))
            if doc["doc_id"] not in scirex_to_s2orc_metadata:
                continue

            doc_abstract = scirex_to_s2orc_metadata[doc["doc_id"]]['abstract']
            assert doc_abstract is None or doc_abstract in abstract_to_annotations, breakpoint()

            query = abstract_to_annotations[str(doc_abstract)]["query"]
            if doc_abstract is None or query == "" or abstract_to_annotations[doc_abstract]["task"] == "":
                continue

            doc_annotations = abstract_to_annotations[doc_abstract]

            if args.use_keyphrases:
                multiquery = construct_multiple_keyword_queries(doc_annotations, keyphrase_keys)
                tldr = doc_annotations["query"]
                query = construct_keyword_query(doc_annotations, keyphrase_keys)
                doc_annotations["query"] = query

            scirex_suggestions = dataset_tags
            s2_id = doc["doc_id"]

            if doc_abstract in backup_files_by_abstract:
                dataset_tags = backup_files_by_abstract[doc_abstract]
                overlap = set(scirex_suggestions).intersection(set(dataset_tags))
                if len(overlap) == 0:
                    p = 0
                    r = 0
                else:
                    p = len(overlap) / len(set(scirex_suggestions))
                    r = len(overlap) / len(set(dataset_tags))
                precisions.append(p)
                recalls.append(r)
                exact_match.append(int(set(scirex_suggestions) == set(dataset_tags)))
            else:
                if len(dataset_tags) == 0:
                    no_datasets_tagged_counter += 1
                    print("All paragraphs:\n\n" + "\n".join(paragraphs))
                else:
                    sections_with_matched_variants = []
                    for p in paragraphs:
                        variant_in_paragraph = False
                        for var in matched_variants:
                            if var in p:
                                variant_in_paragraph = True
                                break
                        if variant_in_paragraph:
                            sections_with_matched_variants.append(p)
                    if len(sections_with_matched_variants) > 0:
                        print(f"Possible Result Sections:\n\n" + "\n".join(sections_with_matched_variants))
                    else:
                        print("All paragraphs:\n\n" + "\n".join(paragraphs))

                s2_link = f"https://api.semanticscholar.org/{s2_id}"
                print(f"s2 link:\n{s2_link}\n")

                dataset_tags_formatted = ", ".join(dataset_tags)
                start = time.perf_counter()
                print(f"\n\nSuggested dataset tags:\n{dataset_tags_formatted}")
                print(f"SciREX-extracted dataset names:\n{datasets_labeled}\n\n")

                print("Corrections needed? (Y/n)")
                breakpoint()
                corrections_needed = input_y_n()
                if corrections_needed:
                    print("Enter comma-separated list of datasets:\n")
                    dataset_tags = input_csv(dataset_ids)

                elapsed = time.perf_counter() - start
                print(f"Example took {round(elapsed, 2)} seconds. Goal is 60 per example.")

            if not args.use_keyphrases:
                multiquery = [query]
            else:
                query = tldr

            for m in multiquery:
                multiquery_rows.append({'Original Query': query, 'Keywords': m.lower()})

            if args.keep_queries_with_field is not None:
                keyphrase_field = doc_annotations.get(args.keep_queries_with_field, "").strip()
                if keyphrase_field == "" or keyphrase_field == "N/A":
                    continue

            query_writer.write(query + "\n")
            year = scirex_to_s2orc_metadata[doc["doc_id"]]["year"]
            query_id = "_".join(query.split())
            for dataset in list(set(dataset_tags)):
                docid = "_".join(dataset.split())
                tsv_writer.writerow([query_id, "Q0", docid, "1"])
            num_rows_written += 1

            row = {"documents": sorted(dataset_tags), "year": year, "keyphrase_query": construct_keyword_query(doc_annotations, keyphrase_keys)}
            row.update(doc_annotations)
            backup_writer.write(row)
            queries_and_datasets.append(row)

    print(f"Average Precision of SciREX tags: {round(np.mean(precisions), 3)}")
    print(f"Average Recall of SciREX tags: {round(np.mean(recalls), 3)}")
    print(f"Exact match accuracy of SciREX tags: {round(np.mean(exact_match), 3)}")

    backup_writer.close()
    query_writer.close()
    relevance_file.close()


    multiquery_writer = jsonlines.Writer(open(args.output_multiqueries_file, 'w'))
    multiquery_writer.write_all(multiquery_rows)
    multiquery_writer.close()

    print(f"Wrote {num_rows_written} test documents to {args.output_relevance_file}")

    json.dump(queries_and_datasets, open(args.output_combined_file, 'w'))
    print(f"Wrote {num_rows_written} test documents to {args.output_combined_file}")


    # Fields:
    # 'paper_id', 'title', 'authors', 'abstract', 'year'

    # main(scirex_documents, range_to_label, args.out_labels_directory, range_string)
