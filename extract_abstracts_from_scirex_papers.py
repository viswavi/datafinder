'''
python extract_abstracts_from_scirex_papers.py \
    --scirex-to-s2orc-metadata-file /home/vijayv/pickle_backups/scirex_id_to_s2orc_metadata.pkl \
    --tldr-input-file scirex_abstracts.temp
'''

import argparse
import pickle

def transformed_document(doc):
    return doc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scirex-to-s2orc-metadata-file', type=str, help="Pickle file containing mapping from SciREX paper IDs to S2ORC metadata")
    parser.add_argument('--tldr-input-file', type=str, help="Line-separated values file containing paper abstracts")
    args = parser.parse_args()

    tldr_writer = open(args.tldr_input_file, 'w')

    scirex_to_s2orc_mapping = pickle.load(open(args.scirex_to_s2orc_metadata_file, 'rb'))
    abstracts = []
    for scirex_id in sorted(scirex_to_s2orc_mapping.keys()):
        abstract = scirex_to_s2orc_mapping[scirex_id]["abstract"]
        tldr_writer.write(f"{abstract} <|TLDR|> .\n")
    tldr_writer.close()