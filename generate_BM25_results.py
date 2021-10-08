'''
python generate_BM25_results.py --output-file data/test/retrieved_documents.trec
'''

from pyserini.search import SimpleSearcher

import argparse
import jsonlines
import csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-set", type=str, default="data/test/test_dataset_collection.jsonl", help="Test collection of queries and documents")
    parser.add_argument('--output-file', type=str, default="data/test/retrieved_documents.trec", help="Retrieval file, in TREC format")
    parser.add_argument('--anserini-index', type=str, default="indexes/dataset_collection_jsonl")
    parser.add_argument('--remove-stopwords', action="store_true")
    parser.add_argument('--remove-function-words', action="store_true")
    args = parser.parse_args()

    searcher = SimpleSearcher(args.anserini_index)

    with open(args.output_file, 'w') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(["QueryID", "Q0", "DocID", "Rank", "Score", "RunID"])

        for row in jsonlines.open(args.test_set):
            query = "_".join(row["query"].split())
            if args.remove_stopwords:
                raise NotImplementedError
            if args.remove_function_words:
                raise NotImplementedError
            hits = searcher.search(query)
            for rank, hit in enumerate(hits):
                docid = "_".join(hit.docid.split())
                tsv_writer.write([query, "Q0", docid, str(rank+1), str(hit.score), "run-1"])
        tsv_writer.close()
