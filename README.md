# dataset-recommendation

## Requirements
```
pytorch >= 1.8.2
```

```
pip install paperswithcode-client
pip install pyserini
pip install tevatron
conda install faiss
```

```
git submodule add https://github.com/castorini/anserini
cd anserini
cd tools/eval && tar xvfz trec_eval.9.0.4.tar.gz && cd trec_eval.9.0.4 && make && cd ../../..
cd tools/eval/ndeval && make && cd ../../../..
```



## Processed Dataset
Found in `data/`. Both training and test data contain "tldr", "positives", and "year" for each query. The training set contains other metadata (such as hard negatives and detailed metadata about the paper we used to extract the query).

## Data Preprocessing
Download and untar data from https://github.com/allenai/SciREX/blob/master/scirex_dataset/release_data.tar.gz.
### Prepare Search Corpus
Download and unzip the `datasets` data from https://github.com/paperswithcode/paperswithcode-data, and place into `data/`.

`python data_processing/build_search_corpus/generate_datasets_collection.py --exclude-abstract --exclude-full-text --output-file data/dataset_search_collection.jsonl`

### Prepare Test Data

mkdir intermediate_data
export PICKLES_DIRECTORY=intermediate_data
./add_tldrs_to_scirex_abstracts.sh

```
python data-processing/test_data/convert_scirex.py \
    --scirex-directory $SCIREX_PATH/scirex_dataset/release_data/ \
    --dataset-search-collection data/dataset_search_collection.jsonl \
    --datasets-file datasets.json \
    --scirex-to-s2orc-metadata-file $PICKLES_DIRECTORY/scirex_id_to_s2orc_metadata_with_tldrs.pkl \
    --output-relevance-file data/test_dataset_collection.qrels \
    --output-queries-file test_queries.csv \
    --output-combined-file data/test_data.jsonl \
    --training-set-documents tagged_dataset_positives.jsonl \
    --bad-query-filter-map bad_tldrs_mapping.json
```

### Prepare Training Data

See [training data preparation instructions](data_processing/train_data/README.md).

## Training
See [biencoder training instructions](retrieval/biencoder/tevatron_scripts/README.md#Training).

## Retrieval

### BM25
```
python retrieval/bm25/generate_results.py \
--anserini-index anserini_search_collections/dataset_search_collection_no_abstracts_or_paper_text_jsonl \
--output-file retrieved_documents_bm25.trec \
--results-limit 5
```

### k-NN (TF-IDF features)
```
python retrieval/knn/generate_results.py \
    --remove-punctuation \
    --remove-stopwords \
    --training-set data/train_data.jsonl \
    --training-tldrs data/train_tldrs.hypo \
    --search-collection anserini_search_collections/dataset_search_collection_no_abstracts_or_paper_text/documents.jsonl \
    --output-file retrieved_documents_knn_tfidf.trec \
    --vectorizer-type tfidf \
    --results-limit 5
```

### k-NN (BERT features)
```
python retrieval/knn/generate_results.py \
    --remove-punctuation \
    --remove-stopwords \
    --query-metadata data/test/scirex_queries_and_datasets.json \
    --training-set data/train_data.jsonl \
    --training-tldrs data/train_tldrs.hypo \
    --search-collection anserini_search_collections/dataset_search_collection_no_abstracts_or_paper_text/documents.jsonl \
    --output-file data/test/retrieved_documents_knn_exact_bert.trec \
    --vectorizer-type bert \
    --results-limit 5
```

### Bi-Encoder (Tevatron)
See [biencoder retrieval instructions](retrieval/biencoder/tevatron_scripts/README.md#Retrieval).


## Evaluate results
### Core Metrics
```
GOLD_FILE=data/test_dataset_collection.qrels
# Set RETRIEVAL_OUTPUT to the desired file. Example provided:
RETRIEVAL_OUTPUT=data/test/retrieved_documents_knn_exact_bert.trec

./anserini/tools/eval/trec_eval.9.0.4/trec_eval \
    -c \
    -m P.5 \
    -m recall.5 \
    -m map \
    -m recip_rank \
    $GOLD_FILE \
    $RETRIEVAL_OUTPUT
```

### Bucketing by Dataset Frequency
```
GOLD_FILE=data/test_dataset_collection.qrels
# Set RETRIEVAL_OUTPUT to the desired file. Example provided:
RETRIEVAL_OUTPUT=data/test/retrieved_documents_knn_exact_bert.trec

python data_analysis/evaluate_dataset_recall_buckets.py $GOLD_FILE $RETRIEVAL_OUTPUT
```

## Reproducing Experiments
### Labeling tool:
This tool was used to validate the quality of labels in our training set:
`python data_processing/train_data/label_dataset_sentences.py --labeler-name <your name> --range-to-label 1,200`.

