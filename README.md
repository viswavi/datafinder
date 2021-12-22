# dataset-recommendation

## Requirements
```
pytorch >= 1.8.2
```

```
pip install pyserini
pip install tevatron
conda install faiss
```


## Processed Dataset
Found in `data/`. Both training and test data contain "tldr", "positives", and "year" for each query. The training set contains other metadata (such as hard negatives and detailed metadata about the paper we used to extract the query).

## Data Preprocessing
Download and untar data from https://github.com/allenai/SciREX/blob/master/scirex_dataset/release_data.tar.gz
### Prepare Search Corpus
`python generate_datasets_collection.py --exclude-abstract --exclude-full-text --output-file dataset_search_collection.jsonl`

### Prepare Test Data

mkdir intermediate_data
export PICKLES_DIRECTORY=intermediate_data
./add_tldrs_to_scirex_abstracts.sh

```
python convert_scirex.py \
    --scirex-directory SCIREX_PATH/scirex_dataset/release_data/ \
    --dataset-search-collection dataset_search_collection.jsonl \
    --datasets-file datasets.json \
    --scirex-to-s2orc-metadata-file PICKLES_DIRECTORY/scirex_id_to_s2orc_metadata_with_tldrs.pkl \
    --output-relevance-file test_dataset_collection.qrels \
    --output-queries-file test_queries.csv \
    --output-combined-file data/test_data.json \
    --training-set-documents tagged_dataset_positives.jsonl \
    --bad-query-filter-map bad_tldrs_mapping.json
```

### Prepare Training Data

```
python scrub_dataset_mentions_from_tldrs.py train_tldrs.hypo train_tldrs_scrubbed.hypo
```

```
python merge_tagged_datasets.py \
    --combined-file data/train_data.jsonl \
    --dataset-tldrs train_tldrs_scrubbed.hypo \
    --tagged-positives-file tagged_dataset_positives.jsonl \
    --tagged-negatives-file tagged_dataset_negatives.jsonl \
    --negative-mining hard \
    --anserini-index indexes/dataset_search_collection_no_paper_text_jsonl \
    --num-negatives 7
```

## Training


## Reproducing Experiments
### Labeling tool:
`python label_dataset_sentences.py --labeler-name <your name> --range-to-label 1,10`

