import argparse
import csv
import jsonlines
import os
import pickle


def transformed_document(doc):
    return doc

def input_y_n():
    y_n = str(input())
    return y_n.strip() == "" or y_n.strip().lower() == "y"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scirex-directory', type=str, default="/projects/ogma1/vijayv/SciREX/scirex_dataset/release_data/",
                        help="Path to local pickle file containing raw dataset sentences for labeling")
    parser.add_argument('--dataset-search-collection', type=str, default="dataset_search_collection.jsonl", help="Jsonlines file containing all the datasets in our \"search collection\"")
    parser.add_argument('--datasets-file', type=str, default="datasets.json", help="JSON file containing metadata about all datasets on PapersWithCode")
    parser.add_argument('--scirex-to-s2orc-metadata-file', type=str, help="Pickle file containing mapping (with tldrs) from SciREX paper IDs to S2ORC metadata", default="/home/vijayv/pickle_backups/scirex_id_to_s2orc_metadata_with_tldrs.pkl")
    parser.add_argument('--output-relevance-file', type=str, default="data/test/test_dataset_collection.qrels")
    parser.add_argument('--output-queries-file', type=str, default="data/test/test_queries.csv")

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
    tlds = [scirex_to_s2orc_metadata[row]["tldr"] for row in scirex_to_s2orc_metadata]
    relevance_file = open(args.output_relevance_file, 'w')
    tsv_writer = csv.writer(relevance_file, delimiter='\t')
    tsv_writer.writerow(["QueryID", "0", "DocID", "Relevance"])
    query_writer = open(args.output_queries_file, 'w')

    mismatches_cache = {}

    num_rows_written = 0
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
            if doc["doc_id"] not in scirex_to_s2orc_metadata:
                continue
            tldr = scirex_to_s2orc_metadata[doc["doc_id"]]["tldr"]
            if "None <|TLDR|>" in tldr:
                continue
            if len(set(dataset_tags)) != len(set(datasets_labeled)):
                print(f"Possible dataset mismatch found: {list(set(dataset_tags))} vs {list(set(datasets_labeled))}")
                print("Is this mismatch ok? Y/n")
                mismatch_accepted = True
                # mismatch_accepted = input_y_n()
                # mismatches_cache[tuple(tuple(set(dataset_tags)), tuple(set(datasets_labeled)))] = y_n
                if not mismatch_accepted:
                    continue
            query_writer.write(tldr + "\n")
            query_id = "_".join(tldr.split())
            for dataset in dataset_tags:
                docid = "_".join(dataset.split())
                tsv_writer.writerow([query_id, "Q0", docid, "1"])
            num_rows_written += 1

            row = {"query": tldr, "documents": sorted(dataset_tags)}
    query_writer.close()
    relevance_file.close()

    print(f"Wrote {num_rows_written} test documents to {args.output_relevance_file}")
    # pickle.dump(mismatches_cache, open("dataset_mismatches_cache.pkl", 'wb'))

    # Fields:
    # 'paper_id', 'title', 'authors', 'abstract', 'year'

    # main(scirex_documents, range_to_label, args.out_labels_directory, range_string)
