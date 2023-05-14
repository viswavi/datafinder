'''
python generate_knn_results.py \
    --remove-punctuation \
    --remove-stopwords \
    --query-metadata data/test/scirex_queries_and_datasets.json \
    --training-set tagged_dataset_positives.jsonl \
    --training-tldrs tagged_dataset_tldrs.hypo \
    --search-collection dataset_search_collection/documents.jsonl \
    --output-file data/test/retrieved_documents_knn_exact_longer_input_tfidf.trec \
    --vectorizer-type tfidf \
    --knn-aggregator exact_top

python generate_knn_results.py \
    --remove-punctuation \
    --remove-stopwords \
    --query-metadata data/test/scirex_queries_and_datasets.json \
    --training-set tagged_dataset_positives.jsonl \
    --training-tldrs tagged_dataset_tldrs.hypo \
    --search-collection dataset_search_collection/documents.jsonl \
    --output-file data/test/retrieved_documents_knn_weighted.trec \
    --vectorizer-type tfidf \
    --knn-aggregator weighted

python generate_knn_results.py \
    --remove-punctuation \
    --remove-stopwords \
    --query-metadata data/test/scirex_queries_and_datasets.json \
    --training-set tagged_dataset_positives.jsonl \
    --training-tldrs tagged_dataset_tldrs.hypo \
    --search-collection dataset_search_collection/documents.jsonl \
    --output-file data/test/retrieved_documents_knn_exact_bert.trec \
    --vectorizer-type bert \
    --knn-aggregator exact_top
'''

import argparse
from collections import defaultdict
import csv
import json
import jsonlines
import numpy as np
import os
import pickle
import string
import time
from typing import List
from transformers import pipeline

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

def construct_scibert_vectorizer(device=2):
    model = "allenai/scibert_scivocab_uncased"
    feature_extractor = pipeline('feature-extraction', model=model, tokenizer=model, device=device)
    return feature_extractor


def vectorize_text(text_lines, vectorizer, vectorizer_type, batch_size=400):
    start = time.perf_counter()
    if vectorizer_type == "tfidf":
        vectorized_text_sparse = vectorizer.transform(text_lines)
        vectorized_text = np.array(vectorized_text_sparse.todense())
    elif vectorizer_type == "bert":
        bert_vectors = []
        for batch_idx in range(int(np.ceil(len(text_lines) / batch_size))):
            batch_bert_vectors = vectorizer(text_lines[batch_idx*batch_size:(batch_idx+1)*batch_size])
            bert_vectors.extend(batch_bert_vectors)
        vectorized_text = np.array([v[0] for v in bert_vectors])
    else:
        raise ValueError(f"Unsupported vectorizer type supplied: {vectorizer_type}")
    end = time.perf_counter()
    print(f"Vectorizing {len(text_lines)} lines took {round(end-start, 4)} seconds.")
    return vectorized_text

def prepare_training_set(training_set, training_tldrs, vectorizer_type="tfidf", overwrite_cache=True, remove_function_words=False, remove_punctuation=False, lowercase_query=False, remove_stopwords=False):
    TRAINING_SET_CACHE = os.path.join(PICKLE_CACHES_DIR, vectorizer_type + "_vectorized_data.pkl")
    VECTORIZER_CACHE = os.path.join(PICKLE_CACHES_DIR, vectorizer_type + "_vectorizer.pkl")
    final_training_tldrs = []
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
            if len(row["datasets"]) == 0:
                continue
            assert len(row["datasets"]) > 0
            texts.append(text)
            final_training_tldrs.append(tldr)
            dataset_labels.append(row["datasets"])
        if vectorizer_type == "tfidf":
            vectorizer = TfidfVectorizer(min_df=2)
            vectorizer.fit(texts)
        elif vectorizer_type == "bert":
            vectorizer = construct_scibert_vectorizer()
        else:
            raise ValueError(f"vectorizer type {vectorizer_type} unsupported")
        vectorized_texts = vectorize_text(texts, vectorizer, vectorizer_type)
        vectorized_training_data = list(zip(vectorized_texts, dataset_labels))
        pickle.dump(vectorized_training_data, open(TRAINING_SET_CACHE, 'wb'))
        pickle.dump(vectorizer, open(VECTORIZER_CACHE, 'wb'))
    return vectorized_training_data, vectorizer, final_training_tldrs


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
    final_hits = sorted(final_hits, key=lambda result: result.score)
    return final_hits


def knn_search(query_text, query_metadata, dataset_metadata, query_vectors, faiss_index, training_data, training_set_tldrs, combiner="weighted", num_results=4):
    start = time.perf_counter()
    datasets_list = [datasets for _, datasets in training_data]

    if combiner == "exact_top":
        knn_distances, knn_indices = faiss_index.search(np.array(query_vectors), num_results * 10)
    else:
        knn_distances, knn_indices = faiss_index.search(np.array(query_vectors), 10)

    all_hits = []
    for row_idx in range(len(query_vectors)):
        query_meta = query_metadata[row_idx]
        query_year = query_meta["year"]
        text = query_text[row_idx]
        hits = []
        if combiner == "exact_top":
            for i, score in enumerate(knn_distances[row_idx]):
                for d in datasets_list[knn_indices[row_idx][i]]:
                    if not ("year" in dataset_metadata[d] and query_year < dataset_metadata[d]["year"]):
                        hits.append(SearchResult(d, score))
                if len(hits) >= num_results:
                    break
        elif combiner == "weighted":
            dataset_weighted_scores = defaultdict(float)

            max_distance = max(knn_distances[row_idx])
            reverse_normalized_distances = np.array([(max_distance - d) / max_distance for d in knn_distances[row_idx]])
            hits = []
            for idx in range(len(knn_indices[row_idx])):
                score = reverse_normalized_distances[idx]
                for d in datasets_list[idx]:
                    if not ("year" in dataset_metadata[d] and query_year < dataset_metadata[d]["year"]):
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
    parser.add_argument('--query-metadata', type=str, default="data/test/scirex_queries_and_datasets.json")
    parser.add_argument('--output-file', type=str, default="data/test/retrieved_documents.trec", help="Retrieval file, in TREC format")
    parser.add_argument('--search-collection', type=str, default="dataset_search_collection/documents.jsonl")
    parser.add_argument('--remove-function-words', action="store_true")
    parser.add_argument('--remove-punctuation', action="store_true")
    parser.add_argument('--lowercase-query', action="store_true")
    parser.add_argument('--remove-stopwords', action="store_true")
    parser.add_argument('--knn-aggregator', type=str, choices=["exact_top", "weighted"])
    parser.add_argument('--vectorizer-type', type=str, choices=["tfidf", "bert"])
    parser.add_argument('--results-limit', default=None, type=int)
    parser.add_argument('--use-keyphrases', action="store_true")
    args = parser.parse_args()

    training_set = list(jsonlines.open(args.training_set))
    training_set_tldrs = open(args.training_tldrs).read().split("\n")

    query_metadata = json.load(open(args.query_metadata))
    search_collection = list(jsonlines.open(args.search_collection))
    vectorized_training_data, vectorizer, final_training_tldrs = prepare_training_set(training_set,
                                                                training_set_tldrs,
                                                                vectorizer_type=args.vectorizer_type,
                                                                remove_function_words=args.remove_function_words,
                                                                remove_punctuation=args.remove_punctuation,
                                                                lowercase_query=args.lowercase_query,
                                                                remove_stopwords=args.remove_stopwords)

    faiss_index = construct_faiss_index(vectorized_training_data)

    with open(args.output_file, 'w') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(["QueryID", "Q0", "DocID", "Rank", "Score", "RunID"])

        if args.use_keyphrases:
            test_queries = [q["keyphrase_query"] for q in query_metadata]
        else:
            test_queries = [q["query"] for q in query_metadata]
        full_queries = [q["query"] for  q in query_metadata]
        valid_queries = [q["text"] for q in jsonlines.open("tevatron_data/test_queries.jsonl")]




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

        dataset_metadata = {}
        for row in search_collection:
            dataset_metadata[row["id"]] = row

        all_hits = knn_search(test_queries, query_metadata, dataset_metadata, query_vectors, faiss_index, vectorized_training_data, final_training_tldrs, combiner=args.knn_aggregator)
        previous_hits = set()
        for query_idx, hits in enumerate(all_hits):
            if full_queries[query_idx] not in valid_queries:
                continue
            query_id = "_".join(full_queries[query_idx].split())
            for rank, hit in enumerate(hits):
                docid = "_".join(hit.docid.split())
                tsv_writer.writerow([query_id, "Q0", docid, str(rank+1), str(hit.score), "run-1"])
                if args.results_limit is not None and rank + 1 == args.results_limit:
                    break