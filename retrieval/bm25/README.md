# Index construction
## Available search corpora
Prepared search corpora can be found at `anserini_search_collections`. We have included three types of indices which we experimented on:
1. `anserini_search_collections/dataset_search_collection_description_title_only`: dataset descriptions and titles of the dataset's introducing paper
2. `anserini_search_collections/dataset_search_collection_with_paper_abstract`: dataset descriptions, introducing paper titles, and introducing paper abstracts
3. `anserini_search_collections/dataset_search_collection_with_paper_abstract_and_text`: dataset descriptions, introducing paper titles, introducing paper abstracts, and the full text of the introducing paper

In our experiments, we found that #1 was most effective for BM25 retrieval.

## Prepare Your Own Search Corpus
You can alternatively prepare your own search corpus (e.g. `anserini_search_collections/dataset_search_collection_description_title_only`). See the main README at the root of this repository for instructions. This also requires installation of Pyserini.

```
mkdir -p anserini_search_collections/dataset_search_collection_description_title_only
python data_processing/build_search_corpus/configure_search_collection.py \
    --exclude-abstract \
    --exclude-full-text \
    --full-documents data/dataset_search_collection.jsonl \
    --filtered-documents anserini_search_collections/dataset_search_collection_description_title_only/documents.jsonl
```

## Construct Anserini index
Given a search corpus, you must now construct an Anserini index to enable fast retrieval:
```
mkdir -p indexes
python -m pyserini.index -collection JsonCollection \
                         -generator DefaultLuceneDocumentGenerator \
                         -threads 1 \
                         -input anserini_search_collections/dataset_search_collection_description_title_only  \
                         -index indexes/dataset_search_collection_description_title_only \
                         -storePositions -storeDocvectors -storeRaw
```

# BM25 Retrieval
After constructing the Anserini index (found at `indexes/dataset_search_collection_description_title_only`), you can run BM25 retrieval using the `retrieval/bm25/generate_results.py` script:
```
python retrieval/bm25/generate_results.py \
--anserini-index indexes/dataset_search_collection_description_title_only \
--output-file retrieved_documents_bm25.trec \
--query-metadata data/test_data.jsonl \
--search-collection data/dataset_search_collection.jsonl \
--results-limit 5
```