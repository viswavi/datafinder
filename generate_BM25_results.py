'''
python generate_BM25_results.py --anserini-index indexes/dataset_collection_jsonl --output-file data/test/retrieved_documents_4_limit.trec --results-limit 4

python generate_BM25_results.py --anserini-index indexes/dataset_search_collection_no_abstracts_or_paper_text_jsonl --output-file data/test/retrieved_documents_query_filtered_4_limit.trec --results-limit 4


python generate_BM25_results.py \
    --remove-punctuation \
    --remove-stopwords \
    --anserini-index indexes/dataset_collection_jsonl \
    --output-file data/test/retrieved_documents_query_stopwords_and_punctuation_removed.trec \
    --search-collection dataset_search_collection/documents.jsonl
'''
from nltk.corpus import stopwords
from pyserini.search import SimpleSearcher
import spacy

import argparse
import csv
import json
import jsonlines
import string

nlp = spacy.load("en_core_web_sm")

def remove_punctuation(st):
    return " ".join(''.join(' ' if c in string.punctuation else c for c in st).split())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-set", type=str, default="test_dataset_collection.jsonl", help="Test collection of queries and documents")
    parser.add_argument('--test_queries', type=str, default="test_queries.csv", help="List of newline-delimited queries")
    parser.add_argument('--query-metadata', type=str, default="test_data.json")
    parser.add_argument('--output-file', type=str, default="retrieved_documents.trec", help="Retrieval file, in TREC format")
    parser.add_argument('--anserini-index', type=str, default="indexes/dataset_collection_jsonl")
    parser.add_argument('--search-collection', type=str, default="dataset_search_collection/documents.jsonl")
    parser.add_argument('--remove-function-words', action="store_true")
    parser.add_argument('--remove-punctuation', action="store_true")
    parser.add_argument('--lowercase-query', action="store_true")
    parser.add_argument('--remove-stopwords', action="store_true")
    parser.add_argument('--results-limit', default=None, type=int)
    args = parser.parse_args()

    dataset_metadata = {}
    for row in jsonlines.open(args.search_collection):
        dataset_metadata[row["id"]] = row

    query_metadata = {}
    for row in json.load(open(args.query_metadata)):
        query_metadata[row["tldr"]] = row

    searcher = SimpleSearcher(args.anserini_index)
    searcher.set_bm25(k1=0.8, b=0.4)

    with open(args.output_file, 'w') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(["QueryID", "Q0", "DocID", "Rank", "Score", "RunID"])

        test_queries = [row for row in open(args.test_queries).read().split("\n") if len(row.split()) != 0] 
        for query in test_queries:
            query_year = query_metadata[query]["year"]
            query_id = "_".join(query.split())
            if args.remove_function_words:
                st = ""
                spacy_doc = nlp(query)
                for token in spacy_doc:
                    if token.pos_ in ["NOUN", "ADJ", "VERB"]:
                        st = st + " " + token.string 
                query = " ".join(st.split())
            if args.remove_punctuation:
                query = remove_punctuation(query)
            if args.lowercase_query:
                query = query.lower()
            if args.remove_stopwords:
                stopwords_without_punctuation = [remove_punctuation(stopword) for stopword in stopwords.words('english')]
                non_stopwords = []
                for w in query.split():
                    if w.lower() not in stopwords.words('english'):
                        non_stopwords.append(w)
                query = " ".join(non_stopwords)
            max_results_retrieved = 50 if args.results_limit is None else 5*args.results_limit
            hits = searcher.search(query, k=max_results_retrieved)
            previous_hits = set()
            num_actual_hits = 0
            for rank, hit in enumerate(hits):
                docid = "_".join(hit.docid.split())
                if docid in previous_hits or (dataset_metadata[hit.docid].get("year", None) is not None and query_year < dataset_metadata[hit.docid]["year"]):
                    rank -= 1
                    continue
                else:
                    previous_hits.add(docid)
                num_actual_hits += 1
                tsv_writer.writerow([query_id, "Q0", docid, str(rank+1), str(hit.score), "run-1"])
                if args.results_limit is not None and rank + 1 == args.results_limit:
                    break
            assert num_actual_hits > 0, breakpoint()