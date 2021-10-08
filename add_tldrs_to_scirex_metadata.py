import argparse
import pickle

def transformed_document(doc):
    return doc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scirex-to-s2orc-metadata-file', type=str, help="Pickle file containing mapping from SciREX paper IDs to S2ORC metadata")
    parser.add_argument('--tldr-file', type=str, help="Newline-delimited values file containing TL;DRs")
    parser.add_argument('--new-scirex-to-s2orc-metadata-file', type=str, help="Pickle file containing updated mapping with TL;DRs added")
    args = parser.parse_args()

    tldrs = [line for line in open(args.tldr_file).read().split("\n") if len(line.split()) > 0]
    scirex_to_s2orc_mapping = pickle.load(open(args.scirex_to_s2orc_metadata_file, 'rb'))
    assert len(scirex_to_s2orc_mapping.keys()) == len(tldrs), breakpoint()

    abstracts = []
    for i, scirex_id in enumerate(sorted(scirex_to_s2orc_mapping.keys())):
        scirex_to_s2orc_mapping[scirex_id]["tldr"] = tldrs[i]

    pickle.dump(scirex_to_s2orc_mapping, open(args.new_scirex_to_s2orc_metadata_file, 'wb'))
