'''
python generate_BM25_results.py
'''

from pyserini.search import SimpleSearcher

import argparse
import jsonlines

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-set", type=str, default="data/test/test_dataset_collection.jsonl", help="Test collection of queries and documents")
    parser.add_argument('--output-file', type=str, default="data/test/retrieved_documents.jsonl")
    parser.add_argument('--anserini-index', type=str, default="indexes/dataset_collection_jsonl")
    parser.add_argument('--remove-stopwords', action="store_true")
    parser.add_argument('--remove-function-words', action="store_true")
    args = parser.parse_args()

    searcher = SimpleSearcher(args.anserini_index)

    output_writer = jsonlines.Writer(open(args.output_file, 'w'))
    for row in jsonlines.open(args.test_set):
        query = row["query"]
        if args.remove_stopwords:
            raise NotImplementedError
        if args.remove_function_words:
            raise NotImplementedError
        hits = searcher.search(query)
        output_writer.write({"query": query, "documents": [hit.docid for hit in hits], "scores": [hit.score for hit in hits]})
    output_writer.close()