
"""
Usage:
python convert_scirex_documents_to_system_descriptions.py \
    --raw-dataset-sentences-file raw_dataset_sentences/dataset_sentence_snippets_no_cap_ref_match_variant_filter.pkl \
"""

import argparse
import json
import jsonlines
import os
import pickle
import time

SCIREX_PATH = "SCIREX_PATH/scirex_dataset/release_data/"

def input_string():
    description = str(input())
    return description

def input_y_n():
    y_n = str(input())
    return y_n.strip() == "" or y_n.strip().lower() == "y"

def display_sentence_meta(paper_idx, s2_paper_id, section_words):
    s2_url=f"http://api.semanticscholar.org/{s2_paper_id}"
    print(f"Paper #{paper_idx}\n")
    print(f"S2 URL: {s2_url}")
    paper = "\n\n".join(section_words)
    print(f"Document:\n\n{paper}")

def label_sentences():
    print(f"\nEnter system description (1-3 sentences):")
    description = input_string()
    return description
    
def write_label(label_file, s2orc_id, system_description):
    label_meta = {
                    "s2orc_id": s2orc_id,
                    "system_description": system_description
                 }
    json.dump(label_meta, open(label_file, 'w'))

def read_label_for_display(label_file):
    label = json.load(open(label_file))
    s2orc_id = label["s2orc_id"]
    system_description = label["system_description"]
    return s2orc_id, system_description

def main(scirex_documents, range_to_label, labels_directory, range_string):

    os.makedirs(labels_directory, exist_ok=True)
    
    print(range_string)

    start_timer = time.perf_counter()

    for paper_idx in range_to_label:
        if paper_idx >= len(scirex_documents):
            print("Done!")
            break

        doc_id = scirex_documents[paper_idx]["doc_id"]
        section_words = []
        for [section_start, section_end] in scirex_documents[paper_idx]["sections"]:
            section_words.append(" ".join(scirex_documents[paper_idx]["words"][section_start:section_end]))
        paper_id = str(paper_idx + 1)

        display_sentence_meta(paper_id, doc_id, section_words)
        new_label_file = os.path.join(labels_directory, f"{paper_id}.json")
        if os.path.isfile(new_label_file):
            s2orc_id, system_description = read_label_for_display(new_label_file)
            existing_labels = {"s2orc_id": s2orc_id, "system_description": system_description}
            print(f"Label for document {paper_id} already exists. Current label:\n{json.dumps(existing_labels, indent=2)}")
            print(f"\nOverwrite? ([y]/n)")
            overwrite = input_y_n()
            if not overwrite:
                continue
        else:
            description= label_sentences()
        write_label(new_label_file, doc_id, description)

    end_timer = time.perf_counter()
    elapsed_time = round(end_timer - start_timer, 2)
    print(f"Annotating {len(range_to_label)} examples took {elapsed_time} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--labeler-name', type=str, required=True, help="Please list your Andrew ID.")
    parser.add_argument('--range-to-label', type=str, default="1, 20", help="One-indexed range. Can be either a single index (e.g. \"10\") or an inclusive range (e.g. \"11,20\").")
    parser.add_argument('--out-labels-directory', type=str, default="scirex_descriptions", help="Will create if it doesn't currently exist.")
    parser.add_argument('--scirex-directory', type=str, default=SCIREX_PATH,
                        help="Path to local pickle file containing raw dataset sentences for labeling")
    args = parser.parse_args()

    if "," in args.range_to_label:
        args.range_to_label = f"[{args.range_to_label}]"
    range_to_label = json.loads(args.range_to_label)
    range_str = str(range_to_label) if isinstance(range_to_label, int) else f"range()"

    if isinstance(range_to_label, list):
        assert len(range_to_label) == 2
        range_string = f"Labeling papers in range [{range_to_label[0]}, {range_to_label[1]}]."
        # Create an inclusive zero-indexed range from the inputted pair of indices.
        range_to_label = range(range_to_label[0] - 1, range_to_label[1])
    elif isinstance(range_to_label, int):
        range_string = f"Labeling paper {range_to_label}."
        #Make index zero-indexed instead of one-indexed.
        range_to_label = [range_to_label-1]
    else:
        raise ValueError("Unexpected data type.")

    scirex_documents = []
    for file in os.listdir(args.scirex_directory):
        scirex_split_file = os.path.join(args.scirex_directory, file)
        scirex_documents.extend(list(jsonlines.open(scirex_split_file)))

    main(scirex_documents, range_to_label, args.out_labels_directory, range_string)
