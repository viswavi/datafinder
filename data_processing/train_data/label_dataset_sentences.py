"""
Usage:
python label_dataset_sentences.py --labeler-name NAME \
    --raw-dataset-sentences-file raw_dataset_sentences/dataset_sentence_snippets_no_cap_ref_match_variant_filter.pkl \
    --range-to-label 5,20

"""

import argparse
import json
import os
import pickle
import random
import time

def print_labeling_standard():
    print("Welcome to the Dataset Tagging Tool.\n")
    print("The labeling task is that given a dataset name and a paper, identify whether " + 
          "the paper uses this dataset in its model training, experiments, or evaluation. " +
          "We will show you a list of snippets from a single paper, corresponding to a set of " + 
          "mentions of a given dataset.\nFirst write whether this dataset is used in the the paper. " +
          "Then, if used, list the 1-2 snippets from the paper that most explicitly, unambiguously " +
          "state that the paper uses the dataset.")
    print("If it's not abundantly clear whether the dataset is used or not, enter \"unclear\".")
    print(f"\nIf you wish to go back and update an annotation, Ctrl-C out of this tool, and " +
           "rerun it with the `--range-to-label <example_idx>` argument.")
    time.sleep(1)
    print(f"\n====================\n\n")

def load_dataset_sentences(dataset_snippets_file, seed=0):
    dataset_snippets = pickle.load(open(dataset_snippets_file, 'rb'))
    return dataset_snippets

def input_y_n():
    y_n = str(input())
    return y_n.strip() == "" or y_n.strip().lower() == "y"

def input_y_n_u():
    correct_input=False
    while not correct_input:
        y_n_u = str(input()).strip().lower()
        if y_n_u == "":
            y_n_u = "y"
        if y_n_u in ["y", "n", "unclear"]:
            correct_input=True
        else:
            print(f'Invalid option. Please enter one of ["y", "n", "unclear"].')
    return y_n_u

def input_comma_separated_list():
    correct_input=False
    while not correct_input:
        try:
            salient_indices = str(input())
            str_indices = salient_indices.split(',')
            indices = [int(s.strip()) for s in str_indices]
            correct_input=True
        except:
            print(f'Invalid input. Please enter a comma-separated list of integers.')
    return indices

def display_sentence_meta(paper_idx, s2_paper_id, dataset_name, dataset_variants, dataset_description, candidate_sections):
    s2_url=f"http://api.semanticscholar.org/CorpusID:{s2_paper_id}"
    dataset_variants = ", ".join([f'"{d}"' for d in dataset_variants])
    print(f"Paper #{paper_idx}\n")
    print(f"S2 URL: {s2_url}")
    print(f"Candidate dataset: {dataset_name}\t\tVariants: [{dataset_variants}]\n")
    if dataset_description is not None:
        print(f"Dataset description: {dataset_description}")
    print("----\n\nDocument Sections:\n\n" + "\n".join([f"#{i+1}: \t{section}" for i, section in enumerate(candidate_sections)]))


def label_sentences(has_single_section):
    start_counter = time.perf_counter()
    print(f"\nIs this dataset used in the paper? ([y]/n/unclear)")
    salience = input_y_n_u()

    if salience == "y":
        if has_single_section:
            salient_indices = [0]
        else:
            print(f"\nWhich mentions indicate that the dataset is used? (provide comma-separated list)")
            salient_indices = input_comma_separated_list()
            # Salient indices provided are 1-indexed, so make them 0-indexed.
            salient_indices = [i - 1 for i in salient_indices]
    else:
        salient_indices = []

    end_counter = time.perf_counter()
    print(f"Took {round(end_counter - start_counter, 2)} seconds to label sample!")
    print("===========================\n\n\n")

    # Return salience (yes/no/unclear) and salient snippet indices (1-indexed).
    return salience, salient_indices
    
def write_label(label_file, paper_idx, s2_id, salience, salient_indices, dataset_name, dataset_variants, candidate_sections):
    label_meta = {
                    "paper_idx": paper_idx,
                    "s2_id": s2_id,
                    "salience": salience,
                    "salient_indices": salient_indices,
                    "dataset_name": dataset_name,
                    "dataset_variants": dataset_variants,
                    "candidate_sections": candidate_sections
                 }
    json.dump(label_meta, open(label_file, 'w'))

def read_label_for_display(label_file):
    label = json.load(open(label_file))
    salience = label["salience"]
    salient_indices = label["salient_indices"]
    # Convert indices back into 1-indexed form
    salient_indices = [i+1 for i in salient_indices]

    return salience, salient_indices

def load_dataset_descriptions():
    pwc_datasets_file = "datasets.json"
    dataset_descriptions = {}
    pwc_datasets = json.load(open(pwc_datasets_file))
    for dataset in pwc_datasets:
        dataset_descriptions[dataset["name"]] = dataset["description"]
    return dataset_descriptions

def main(dataset_snippets_file, range_to_label, labels_directory, range_string):
    dataset_sentences = load_dataset_sentences(dataset_snippets_file)
    dataset_descriptions = load_dataset_descriptions()
    os.makedirs(labels_directory, exist_ok=True)
    
    print_labeling_standard()
    print(range_string)

    start_timer = time.perf_counter()

    for paper_idx in range_to_label:
        if paper_idx >= len(dataset_sentences):
            print("Done!")
            break

        dataset_s2_id = dataset_sentences[paper_idx]["paper_id"]
        dataset_name = dataset_sentences[paper_idx]["candidate_dataset"]
        dataset_variants = dataset_sentences[paper_idx]["dataset_variants"]
        dataset_description = dataset_descriptions.get(dataset_name, None)
        candidate_sections = dataset_sentences[paper_idx]["candidate_sections"]
        paper_id = str(paper_idx + 1)

        display_sentence_meta(paper_id, dataset_s2_id, dataset_name, dataset_variants, dataset_description, candidate_sections)
        new_label_file = os.path.join(labels_directory, f"{paper_id}.json")
        if os.path.isfile(new_label_file):
            salience, salient_indices = read_label_for_display(new_label_file)
            existing_labels = {"salience": salience, "salient_indices": salient_indices}
            print(f"Label for document {paper_id} already exists. Current label:\n{json.dumps(existing_labels, indent=2)}")
            print(f"\nOverwrite? ([y]/n)")
            overwrite = input_y_n()
            if not overwrite:
                continue

        has_single_section = len(candidate_sections) == 1
        salience, salient_indices = label_sentences(has_single_section)
        write_label(new_label_file, paper_idx, dataset_s2_id, salience, salient_indices, dataset_name, dataset_variants, candidate_sections)

    end_timer = time.perf_counter()
    elapsed_time = round(end_timer - start_timer, 2)
    print(f"Annotating {len(range_to_label)} examples took {elapsed_time} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--labeler-name', type=str, required=True, help="Please list your Andrew ID.")
    parser.add_argument('--range-to-label', type=str, default="1, 3000", help="One-indexed range. Can be either a single index (e.g. \"10\") or an inclusive range (e.g. \"11,20\").")
    parser.add_argument('--out-labels-directory', type=str, default="dataset_labels", help="Will create if it doesn't currently exist.")
    parser.add_argument('--raw-dataset-sentences-file', type=str, default="raw_dataset_sentences/dataset_sentence_snippets.pkl",
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

    main(args.raw_dataset_sentences_file, range_to_label, args.out_labels_directory, range_string)
