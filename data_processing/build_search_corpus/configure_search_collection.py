'''
python configure_search_collection.py --full-documents dataset_search_collection.jsonl --filtered-documents dataset_search_collection/documents.jsonl
python configure_search_collection.py --exclude-full-text --full-documents dataset_search_collection.jsonl --filtered-documents dataset_search_collection_no_paper_text/documents.jsonl
python configure_search_collection.py --exclude-abstract --exclude-full-text --full-documents dataset_search_collection.jsonl --filtered-documents dataset_search_collection_no_abstracts_or_paper_text/documents.jsonl
'''

import argparse
import jsonlines

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--full-documents', type=str, required=True,
                        help="Path to input file containing full documents")
    parser.add_argument('--filtered-documents', type=str, required=True,
                        help="Path to output file containing filtered documents")
    parser.add_argument('--exclude-abstract', action="store_true")
    parser.add_argument('--exclude-full-text', action="store_true")
    args = parser.parse_args()

    with open(args.filtered_documents, 'w') as out_file:
        writer = jsonlines.Writer(out_file)
        for row in jsonlines.open(args.full_documents):
            if not args.exclude_abstract:
                abstract = "" if row.get("abstract") is None else row["abstract"]
                row["contents"] = row["contents"] + "\t" + abstract
            if not args.exclude_full_text:
                body_text = "" if row.get("body_text") is None else row["body_text"]
                row["contents"] = row["contents"] + "\t" + body_text
            del row["abstract"]
            del row["body_text"]
            writer.write(row)