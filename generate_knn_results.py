'''
python generate_knn_results.py \
    --remove-punctuation \
    --remove-stopwords \
    --training-set tagged_datasets.jsonl \
    --training-tldrs tagged_dataset_tldrs.hypo \
    --search-collection dataset_search_collection/documents.jsonl \
    --output-file data/test/retrieved_documents_knn_exact.trec \
    --vectorizer-type tfidf \
    --knn-aggregator exact_top

python generate_knn_results.py \
    --remove-punctuation \
    --remove-stopwords \
    --training-set tagged_datasets.jsonl \
    --training-tldrs tagged_dataset_tldrs.hypo \
    --search-collection dataset_search_collection/documents.jsonl \
    --output-file data/test/retrieved_documents_knn_weighted.trec \
    --vectorizer-type tfidf \
    --knn-aggregator weighted
'''

import argparse
from collections import defaultdict
import csv
import jsonlines
import numpy as np
import os
import pickle
import string
import time
from typing import List

import faiss
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

nlp = spacy.load("en_core_web_sm")

PICKLE_CACHES_DIR = "/home/vijayv/pickle_backups/dataset_recommendation"

def delete_punctuation(st):
    return " ".join(''.join(' ' if c in string.punctuation else c for c in st).split())

class SearchResult:
    def __init__(self, docid, score):
        self.docid = docid
        self.score = score

def preprocess_text(query, remove_function_words=False, remove_punctuation=False, lowercase_query=False, remove_stopwords=False):
    if remove_function_words:
        st = ""
        spacy_doc = nlp(query)
        for token in spacy_doc:
            if token.pos_ in ["NOUN", "ADJ", "VERB"]:
                st = st + " " + token.string 
        query = " ".join(st.split())
    if remove_punctuation:
        query = delete_punctuation(query)
    if lowercase_query:
        query = query.lower()
    if remove_stopwords:
        if remove_punctuation:
            stopwords_list = [delete_punctuation(stopword) for stopword in stopwords.words('english')]
        else:
            stopwords_list = stopwords.words('english')
        non_stopwords = []
        for w in query.split():
            if w.lower() not in stopwords_list:
                non_stopwords.append(w)
        query = " ".join(non_stopwords)
    return query

def vectorize_text(text_lines, vectorizer, vectorizer_type):
    if vectorizer_type == "tfidf":
        vectorized_text_sparse = vectorizer.transform(text_lines)
        vectorized_text = np.array(vectorized_text_sparse.todense())
        return vectorized_text
    else:
        raise NotImplementedError

def prepare_training_set(training_set, training_tldrs, vectorizer_type="tfidf", overwrite_cache=False, remove_function_words=False, remove_punctuation=False, lowercase_query=False, remove_stopwords=False):
    TRAINING_SET_CACHE = os.path.join(PICKLE_CACHES_DIR, vectorizer_type + "_vectorized_data.pkl")
    VECTORIZER_CACHE = os.path.join(PICKLE_CACHES_DIR, vectorizer_type + "_vectorizer.pkl")
    assert vectorizer_type in ["tfidf", "bert"]
    if os.path.exists(TRAINING_SET_CACHE) and not overwrite_cache:
        vectorized_training_data = pickle.load(open(TRAINING_SET_CACHE, 'rb'))
        vectorizer = pickle.load(open(VECTORIZER_CACHE, 'rb'))
    else:
        texts = []
        dataset_labels = []
        for row, tldr in zip(training_set, training_tldrs):
            text = preprocess_text(tldr,
                                       remove_function_words=remove_function_words,
                                       remove_punctuation=remove_punctuation,
                                       lowercase_query=lowercase_query,
                                       remove_stopwords=remove_stopwords)
            texts.append(text)
            assert len(row["datasets"]) > 0
            dataset_labels.append(row["datasets"])
        if vectorizer_type == "tfidf":
            vectorizer = TfidfVectorizer(min_df=2)
            vectorizer.fit(texts)
            vectorized_texts = vectorize_text(texts, vectorizer, vectorizer_type)
            vectorized_training_data = list(zip(vectorized_texts, dataset_labels))
        else:
            raise ValueError(f"vectorizer type {vectorizer_type} unsupported")
        pickle.dump(vectorized_training_data, open(TRAINING_SET_CACHE, 'wb'))
        pickle.dump(vectorizer, open(VECTORIZER_CACHE, 'wb'))
    return vectorized_training_data, vectorizer


def combine_hits(hits: List[SearchResult]) -> List[SearchResult]:
    final_hits = []
    for h in hits:
        already_retrieved = False
        for i, candidate in enumerate(final_hits):
            if h.docid == candidate.docid:
                final_hits[i].score += h.score
                already_retrieved = True
                break
        if not already_retrieved:
            final_hits.append(h)
    final_hits = sorted(final_hits, key=lambda result: result.score, reverse=True)
    return final_hits

def knn_search(query_text, query_vectors, faiss_index, training_data, combiner="weighted", num_results=4):
    start = time.perf_counter()
    datasets_list = [datasets for _, datasets in training_data]

    if combiner == "exact_top":
        knn_distances, knn_indices = faiss_index.search(np.array(query_vectors), num_results)
    else:
        knn_distances, knn_indices = faiss_index.search(np.array(query_vectors), 10)

    all_hits = []
    for row_idx in range(len(query_vectors)):
        hits = []
        if combiner == "exact_top":
            for i, score in enumerate(knn_distances[row_idx]):
                for d in datasets_list[knn_indices[row_idx][i]]:
                    hits.append(SearchResult(d, score))
        elif combiner == "weighted":
            dataset_weighted_scores = defaultdict(float)

            max_distance = max(knn_distances[row_idx])
            reverse_normalized_distances = np.array([(max_distance - d) / max_distance for d in knn_distances[row_idx]])
            hits = []
            for idx in range(len(knn_indices[row_idx])):
                score = reverse_normalized_distances[idx]
                for d in datasets_list[idx]:
                    dataset_weighted_scores[d] += score
            top_hit_idxs = np.argsort(-np.array(list(dataset_weighted_scores.values())))[:num_results]
            for idx in top_hit_idxs:
                d = list(dataset_weighted_scores.keys())[idx]
                score = list(dataset_weighted_scores.values())[idx]
                hits.append(SearchResult(d, score))
        else:
            raise ValueError("invalid KNN combiner given")
        all_hits.append(combine_hits(hits))
    end = time.perf_counter()
    print(f"Queries took {round(end-start, 4)} seconds for {len(all_hits)} documents")
    return all_hits


def construct_faiss_index(vectorized_training_data):
    vectors = np.array([row[0] for row in vectorized_training_data], dtype=np.float32)
    index = faiss.IndexFlatL2(vectors.shape[1])
    # index = faiss.GpuIndexFlatL2(vectors.shape[0])
    index.add(vectors)
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-set", type=str, default="tagged_datasets.jsonl", help="Training collection of queries and documents")
    parser.add_argument("--training-tldrs", type=str, default="tagged_dataset_tldrs.hypo")
    parser.add_argument("--test-set", type=str, default="data/test/test_dataset_collection.jsonl", help="Test collection of queries and documents")
    parser.add_argument('--test_queries', type=str, default="data/test/test_queries.csv", help="List of newline-delimited queries")
    parser.add_argument('--output-file', type=str, default="data/test/retrieved_documents.trec", help="Retrieval file, in TREC format")
    parser.add_argument('--search-collection', type=str, default="dataset_search_collection/documents.jsonl")
    parser.add_argument('--remove-function-words', action="store_true")
    parser.add_argument('--remove-punctuation', action="store_true")
    parser.add_argument('--lowercase-query', action="store_true")
    parser.add_argument('--remove-stopwords', action="store_true")
    parser.add_argument('--knn-aggregator', type=str, choices=["exact_top", "weighted"])
    parser.add_argument('--vectorizer-type', type=str, choices=["tfidf", "bert"])
    parser.add_argument('--results-limit', default=None, type=int)
    args = parser.parse_args()

    training_set = list(jsonlines.open(args.training_set))
    training_set_tldrs = open(args.training_tldrs).read().split("\n")[:-1]

    vectorized_training_data, vectorizer = prepare_training_set(training_set,
                                                                training_set_tldrs,
                                                                vectorizer_type=args.vectorizer_type,
                                                                overwrite_cache=True,
                                                                remove_function_words=args.remove_function_words,
                                                                remove_punctuation=args.remove_punctuation,
                                                                lowercase_query=args.lowercase_query,
                                                                remove_stopwords=args.remove_stopwords)

    faiss_index = construct_faiss_index(vectorized_training_data)

    with open(args.output_file, 'w') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(["QueryID", "Q0", "DocID", "Rank", "Score", "RunID"])

        test_queries = [row for row in open(args.test_queries).read().split("\n") if len(row.split()) != 0] 
        preprocessed_queries = []
        for query in test_queries:
            query_id = "_".join(query.split())
            query = preprocess_text(query,
                                    remove_function_words=args.remove_function_words,
                                    remove_punctuation=args.remove_punctuation,
                                    lowercase_query=args.lowercase_query,
                                    remove_stopwords=args.remove_stopwords)
            preprocessed_queries.append(query)
        vectorized_queries = vectorize_text(preprocessed_queries, vectorizer, vectorizer_type=args.vectorizer_type)

        query_vectors = []
        for i, query_text in enumerate(test_queries):
            query_vectors.append(vectorized_queries[i])
        query_vectors = np.array(query_vectors, dtype=np.float32)
        all_hits = knn_search(test_queries, query_vectors, faiss_index, vectorized_training_data, combiner=args.knn_aggregator)
        previous_hits = set()
        for query_idx, hits in enumerate(all_hits):
            query = test_queries[query_idx]
            query_id = "_".join(query.split())
            for rank, hit in enumerate(hits):
                docid = "_".join(hit.docid.split())
                tsv_writer.writerow([query_id, "Q0", docid, str(rank+1), str(hit.score), "run-1"])
                if args.results_limit is not None and rank + 1 == args.results_limit:
                    break